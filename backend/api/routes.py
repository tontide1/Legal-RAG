from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.api.schemas import ChatRequest, ChatResponse, ComparisonResponse, UploadResponse
from fastapi.responses import StreamingResponse
import json
import asyncio
import shutil
import os
from backend.config import settings
from backend.core.document_processor import DocumentProcessor
from backend.core.legal_chunker import chunk_markdown, chunks_to_strings

router = APIRouter()

document_processor = DocumentProcessor()


def get_rag_engine():
    from backend.core.rag_engine import RAGEngine

    return RAGEngine.get_instance()


def _normalize_text_response(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text

@router.post("/chat")
async def chat(request: ChatRequest):
    from lightrag import QueryParam

    rag = get_rag_engine()
    print(f"DEBUG: Chat request received. message='{request.message[:20]}...', comparison_mode={request.comparison_mode}, stream={request.stream}")
    
    system_prompt = (
        "STRICT INSTRUCTION: Output ONLY the relevant information. "
        "DO NOT use introductory phrases like 'Dựa trên thông tin được cung cấp...', 'Dưới đây là...', etc. "
        "Directly provide the answer based on the context."
    )
    
    full_query = f"{request.message}\n\n{system_prompt}"
    
    if not request.stream:
        try:
            if request.comparison_mode:
                naive_response = await rag.aquery(full_query, param=QueryParam(mode="naive"))
                hybrid_response = await rag.aquery(full_query, param=QueryParam(mode="hybrid"))
                return ComparisonResponse(
                    naive=ChatResponse(response=naive_response, mode="naive"),
                    hybrid=ChatResponse(response=hybrid_response, mode="hybrid")
                )
            else:
                response = await rag.aquery(full_query, param=QueryParam(mode="hybrid"))
                return ChatResponse(response=response, mode="hybrid")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Streaming Implementation
    async def event_generator():
        queue = asyncio.Queue()
        pending_tasks = set()

        async def stream_wrapper(gen_func, mode):
            try:
                await queue.put(f"data: {json.dumps({'type': 'start', 'mode': mode})}\n\n")
                generator = await gen_func
                emitted = False
                if hasattr(generator, '__aiter__'):
                    async for chunk in generator:
                        normalized_chunk = _normalize_text_response(chunk)
                        if not normalized_chunk:
                            continue
                        emitted = True
                        await queue.put(f"data: {json.dumps({'type': 'chunk', 'mode': mode, 'content': normalized_chunk})}\n\n")
                else:
                    normalized_result = _normalize_text_response(generator)
                    if normalized_result:
                        emitted = True
                        await queue.put(f"data: {json.dumps({'type': 'chunk', 'mode': mode, 'content': normalized_result})}\n\n")

                if not emitted:
                    fallback = await rag.aquery(full_query, param=QueryParam(mode=mode))
                    normalized_fallback = _normalize_text_response(fallback)
                    if not normalized_fallback:
                        raise RuntimeError(f"{mode} query returned no content.")
                    await queue.put(f"data: {json.dumps({'type': 'chunk', 'mode': mode, 'content': normalized_fallback})}\n\n")
            except Exception as e:
                print(f"STREAM ERROR ({mode}): {str(e)}")
                await queue.put(f"data: {json.dumps({'type': 'error', 'mode': mode, 'message': str(e)})}\n\n")

        try:
            if request.comparison_mode:
                # Start both in parallel
                t1 = asyncio.create_task(stream_wrapper(rag.aquery(full_query, param=QueryParam(mode="naive", stream=True)), "naive"))
                t2 = asyncio.create_task(stream_wrapper(rag.aquery(full_query, param=QueryParam(mode="hybrid", stream=True)), "hybrid"))
                pending_tasks.update([t1, t2])
                
                while pending_tasks:
                    # Wait for items in queue or for tasks to finish
                    while not queue.empty():
                        yield await queue.get()
                    
                    done, pending_tasks = await asyncio.wait(pending_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                    
                    # Yield any new items added during wait
                    while not queue.empty():
                        yield await queue.get()
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            else:
                # Standard single stream
                generator = await rag.aquery(full_query, param=QueryParam(mode="hybrid", stream=True))
                emitted = False
                if hasattr(generator, '__aiter__'):
                    async for chunk in generator:
                        normalized_chunk = _normalize_text_response(chunk)
                        if not normalized_chunk:
                            continue
                        emitted = True
                        yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': normalized_chunk})}\n\n"
                else:
                    normalized_result = _normalize_text_response(generator)
                    if normalized_result:
                        emitted = True
                        yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': normalized_result})}\n\n"
                if not emitted:
                    fallback = await rag.aquery(full_query, param=QueryParam(mode="hybrid"))
                    normalized_fallback = _normalize_text_response(fallback)
                    if not normalized_fallback:
                        raise RuntimeError("Hybrid query returned no content.")
                    yield f"data: {json.dumps({'type': 'chunk', 'mode': 'hybrid', 'content': normalized_fallback})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/documents")
async def list_documents():
    rag = get_rag_engine()
    try:
        # Get documents from doc_status storage
        # Use get_docs_paginated to fetch all documents
        docs_tuple, _ = await rag.doc_status.get_docs_paginated()
        
        # Result list
        result = []
        for doc_id, status_obj in docs_tuple:
            # status_obj is a DocProcessingStatus object
            status_str = "unknown"
            if hasattr(status_obj.status, "value"):
                status_str = status_obj.status.value
            elif isinstance(status_obj.status, str):
                status_str = status_obj.status
                
            result.append({
                "id": doc_id,
                "status": status_str,
                "source": status_obj.file_path or "unknown",
                "content_summary": (status_obj.content_summary[:100] + "...") if status_obj.content_summary else ""
            })
        return result
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    lower_filename = filename.lower()

    if not lower_filename.endswith(".pdf") and not lower_filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    file_path = os.path.join(settings.LIGHTRAG_WORKING_DIR, filename)
    os.makedirs(settings.LIGHTRAG_WORKING_DIR, exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        rag = get_rag_engine()
        content = await document_processor.extract_text(file_path)

        if not content.strip():
            raise ValueError("File is empty or no text could be extracted")

        # Chunk by legal structure (Điều/Khoản/Điểm) instead of plain double-newline
        chunks = chunk_markdown(content, source_file=filename)
        chunk_texts = chunks_to_strings(chunks)

        if not chunk_texts:
            raise ValueError("No valid chunks could be extracted from the document")

        print(f"[INFO] {filename}: {len(chunk_texts)} legal chunks produced.")
        await rag.ainsert(chunk_texts, file_paths=[filename] * len(chunk_texts))
            
        return UploadResponse(
            filename=filename,
            status="success",
            message=(
                f"File indexed with the configured embeddings "
                f"({len(content)} characters, {len(chunk_texts)} legal chunks)"
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index file: {str(e)}")
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

@router.get("/health")
async def health():
    return {"status": "healthy"}

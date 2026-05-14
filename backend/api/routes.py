from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.api.schemas import ChatRequest, ChatResponse, ComparisonResponse, UploadResponse
from fastapi.responses import StreamingResponse
import json
import asyncio
from lightrag import LightRAG, QueryParam
from backend.core.rag_engine import RAGEngine
import shutil
import os
from backend.config import settings

router = APIRouter()

from typing import Union


def _normalize_text_response(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text

@router.post("/chat")
async def chat(request: ChatRequest):
    rag = RAGEngine.get_instance()
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
    rag = RAGEngine.get_instance()
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
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        rag = RAGEngine.get_instance()
        
        if lower_filename.endswith(".pdf"):
            from backend.core.llm_services import qwen_vl_parse_pdf
            content = await qwen_vl_parse_pdf(file_path)
        else:
            # Assume TXT
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            raise ValueError("File is empty or no text could be extracted")

        await rag.ainsert(content, file_paths=[filename])
            
        return UploadResponse(
            filename=filename,
            status="success",
            message=(
                "PDF parsed by Qwen 3 VL and indexed with the configured embeddings "
                f"({len(content)} characters)"
                if lower_filename.endswith(".pdf")
                else f"File indexed with the configured embeddings ({len(content)} characters)"
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index file: {str(e)}")

@router.get("/health")
async def health():
    return {"status": "healthy"}

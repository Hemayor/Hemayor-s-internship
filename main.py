from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import json
import os
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

# 导入本地模型和Agent
from agents import search_agent, qa_agent, summary_agent, all_paragraphs

app = FastAPI(
    title="法律智能问答系统",
    description="基于本地大模型的法律智能问答API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 数据模型
class Message(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    topk: Optional[int] = 3  # 检索数量

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    retrieved_texts: List[Dict[str, Any]]  # 检索结果
    summary: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    topk: Optional[int] = 3

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str

class LawDetailRequest(BaseModel):
    text: str
    index: int

# 模拟对话历史存储（在实际应用中应该使用数据库）
conversations = {}

def generate_law_pdf(law_name, article_num, content):
    """生成法条PDF文件"""
    try:
        # 确保PDF目录存在
        pdf_dir = "static/pdfs"
        os.makedirs(pdf_dir, exist_ok=True)
        
        # 生成文件名
        filename = f"{law_name}_{article_num or 'general'}.pdf"
        filepath = os.path.join(pdf_dir, filename)
        
        # 创建PDF文档
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        
        # 获取样式
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=1  # 居中
        )
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leading=18
        )
        
        # 添加标题
        title_text = f"《{law_name}》"
        if article_num:
            title_text += f"第{article_num}条"
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 20))
        
        # 添加内容
        story.append(Paragraph(content, content_style))
        
        # 构建PDF
        doc.build(story)
        
        return filepath
    except Exception as e:
        print(f"生成PDF失败: {e}")
        return None

def extract_law_info(text):
    """从法条文本中提取法律信息"""
    # 尝试提取法律名称和条款号
    law_patterns = [
        r'《([^》]+)》第(\d+)条',
        r'《([^》]+)》第(\d+)款',
        r'《([^》]+)》第(\d+)项',
        r'《([^》]+)》第(\d+)章',
        r'《([^》]+)》第(\d+)节'
    ]
    
    for pattern in law_patterns:
        match = re.search(pattern, text)
        if match:
            law_name = match.group(1)
            article_num = match.group(2)
            return {
                "law_name": law_name,
                "article_num": article_num,
                "full_text": text,
                "has_detail": True
            }
    
    # 如果没有找到具体条款，尝试提取法律名称
    law_name_pattern = r'《([^》]+)》'
    match = re.search(law_name_pattern, text)
    if match:
        law_name = match.group(1)
        return {
            "law_name": law_name,
            "article_num": None,
            "full_text": text,
            "has_detail": False
        }
    
    return {
        "law_name": "未知法律",
        "article_num": None,
        "full_text": text,
        "has_detail": False
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天API接口 - 集成本地模型"""
    try:
        # 生成或获取对话ID
        conversation_id = request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化对话历史
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # 添加用户消息
        user_message = Message(
            role="user",
            content=request.message,
            timestamp=datetime.now()
        )
        conversations[conversation_id].append(user_message)
        
        # 1. 检索相关法条
        topk = request.topk or 3
        retrieved_texts = search_agent(request.message, topk)
        
        # 2. 获取历史对话（最近3轮）
        history = []
        if len(conversations[conversation_id]) > 1:
            # 获取之前的对话历史
            for i in range(0, len(conversations[conversation_id]) - 1, 2):
                if i + 1 < len(conversations[conversation_id]):
                    history.append({
                        "user": conversations[conversation_id][i].content,
                        "assistant": conversations[conversation_id][i + 1].content
                    })
        
        # 3. 调用本地大模型生成回答
        ai_response = qa_agent(request.message, retrieved_texts, history)
        
        # 4. 生成总结
        summary = summary_agent([ai_response])
        
        # 添加AI响应
        assistant_message = Message(
            role="assistant",
            content=ai_response,
            timestamp=datetime.now()
        )
        conversations[conversation_id].append(assistant_message)
        
        # 格式化检索结果，添加法条信息
        formatted_retrieved = []
        for i, (text, score) in enumerate(retrieved_texts):
            law_info = extract_law_info(text)
            formatted_retrieved.append({
                "text": text,
                "score": float(score),
                "index": i + 1,
                "law_info": law_info
            })
        
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            retrieved_texts=formatted_retrieved,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/law-detail")
async def get_law_detail(request: LawDetailRequest):
    """获取法条详细信息"""
    try:
        law_info = extract_law_info(request.text)
        
        # 生成PDF文件名
        pdf_filename = f"{law_info['law_name']}_{law_info['article_num'] or 'general'}.pdf"
        
        return {
            "law_info": law_info,
            "pdf_filename": pdf_filename,
            "has_pdf": os.path.exists(f"static/pdfs/{pdf_filename}"),
            "full_text": request.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-pdf/{filename}")
async def download_pdf(filename: str):
    """下载法条PDF文件"""
    try:
        pdf_path = f"static/pdfs/{filename}"
        if os.path.exists(pdf_path):
            return FileResponse(pdf_path, filename=filename)
        else:
            raise HTTPException(status_code=404, detail="PDF文件不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-pdf")
async def generate_pdf(request: LawDetailRequest):
    """生成并下载法条PDF文件"""
    try:
        law_info = extract_law_info(request.text)
        
        # 生成PDF
        pdf_path = generate_law_pdf(
            law_info['law_name'], 
            law_info['article_num'], 
            request.text
        )
        
        if pdf_path and os.path.exists(pdf_path):
            filename = os.path.basename(pdf_path)
            return FileResponse(pdf_path, filename=filename)
        else:
            raise HTTPException(status_code=500, detail="PDF生成失败")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/law-text/{law_name}/{article_num}")
async def get_law_text(law_name: str, article_num: str):
    """获取法条全文"""
    try:
        # 这里可以根据law_name和article_num从数据库中获取完整法条文本
        # 目前返回模拟数据
        return {
            "law_name": law_name,
            "article_num": article_num,
            "full_text": f"《{law_name}》第{article_num}条的完整内容...",
            "has_pdf": os.path.exists(f"static/pdfs/{law_name}_{article_num}.pdf")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """检索API接口"""
    try:
        topk = request.topk or 3
        retrieved_texts = search_agent(request.query, topk)
        
        results = []
        for i, (text, score) in enumerate(retrieved_texts):
            law_info = extract_law_info(text)
            results.append({
                "text": text,
                "score": float(score),
                "index": i + 1,
                "law_info": law_info
            })
        
        return SearchResponse(
            results=results,
            query=request.query
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取特定对话的历史记录"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除特定对话"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    try:
        # 测试检索功能
        test_query = "测试"
        search_agent(test_query, 1)
        return {
            "status": "healthy",
            "message": "系统运行正常",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统异常: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 
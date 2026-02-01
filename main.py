from __future__ import annotations

from fastapi import FastAPI

from demo import milvus

app = FastAPI()
app.include_router(milvus.router)
app.add_event_handler("startup", milvus.startup)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


'''
功能一：
快速复用历史问题

1.做一个导入接口，从RocketMQ去拉取消息嵌入Milvus向量库，批量插入审计问题
考虑好Collection的id、向量和元数据

2.做一个检索接口，用户输入一段话，根据用户的内容去做混合检索
输入预处理，去噪
向量检索
业务过滤
重排序
只返回 10 条低于阈值的剔除


'''


'''
智能底稿导入
用户输入一段文件，根据用户的信息去整理，并填写好表单

1. POST /manuscript/import                                                                                                                                                                             
     上传文件 + 项目/审计单位等上下文，返回 jobId                                                                                                                                                        
2. GET /manuscript/import/{jobId}/status                                                                                                                                                               
     返回解析进度、是否完成                                                                                                                                                                              
3. GET /manuscript/import/{jobId}/preview                                                                                                                                                              
     返回结构化结果 + 置信度 + 异常                                                                                                                                                                      
4. POST /manuscript/import/{jobId}/confirm                                                                                                                                                             
     用户确认/修正后入库                                                                                                                                                                                 
5. POST /manuscript/import/{jobId}/cancel                                                                                                                                                              
     终止导入或清理临时数据 

'''




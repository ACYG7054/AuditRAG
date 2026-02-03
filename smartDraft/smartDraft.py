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
# 说明：该文件目前仅保留接口说明文档，不包含可执行逻辑。

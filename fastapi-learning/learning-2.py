from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

profile = {"heroTitle":"关于我", "heroSubtitle":"项目、创意、灵感、心得、我的作品"}

@app.get("/api/profile")
def get_profile():
    return profile

#请求体说明，定义了在调用/api/analyze api的时候请求体必须是什么样的类型
class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    return {
        "text":req.text,
        "score":0.5,
        "lable":"偏平淡",
        "pinyin":"大大双"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("learning-2:app", host="127.0.0.1", port=8000, reload=True)
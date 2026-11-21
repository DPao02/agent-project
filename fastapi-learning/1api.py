from api.book import api_book
from api.cbs import api_cbs
from api.zz import api_zz
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from pathlib import Path

#实例化
app = FastAPI()

#设置静态文件路由
UPIMG_DIR = Path(__file__).resolve().parent / "upimg"
app.mount("/upimg", StaticFiles(directory=UPIMG_DIR), name="upimg")#输入/upimg开头的都去找upimg目录下的文件

app.include_router(api_book, prefix="/book", tags=["图书接口",])#路由分发，
                                                                #prefix以book开头都去找api_book模块的路由
                                                                #tags是标签，用于分类路由

app.include_router(api_cbs, prefix="/cbs", tags=["出版社接口",])

app.include_router(api_zz, prefix="/zz", tags=["作者接口",])


@app.get("/")
async def read_root():
    return {"message": "Hello World"}

#request参数
@app.get("/get_test")
async def get_test(request: Request):
    text_msg = request.query_params
    print(text_msg)
    return {"message": text_msg}

#post参数
@app.post("/post_test")
async def post_test(request: Request):
    text_msg = await request.json()
    print(text_msg)
    return {"message": text_msg}

if __name__ == "__main__":
    uvicorn.run("1api:app", host="127.0.0.1", port=8000)

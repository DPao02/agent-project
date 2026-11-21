from fastapi import FastAPI

#web服务器
import uvicorn

#实例化FastAPI类
app = FastAPI()

@app.get('/')
async def read_root():
    return {"message": "Hello pp"}

@app.get("/get")
async def read_get():
    return {"method": "get方法"}

@app.get("/post")
def read_post():
    return {"method": "post方法"}

@app.get("/put")
def read_put():
    return {"method": "put方法"}

@app.get("/delete")
def read_delete():
    return {"method": "delete方法"}


@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.get('/hello')
async def read_hello():
    return {"message": "Hello FastAPI"}

if __name__ == "__main__":
    uvicorn.run("learning-1:app", host="127.0.0.1", port=8080, reload=True)

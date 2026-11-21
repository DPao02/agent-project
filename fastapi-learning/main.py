from fastapi import FastAPI

#创建fastapi实例
app = FastAPI()

#定义路由（定义api接口）————》该函数的返回值表示api接口返回的数据，接口的访问路径为/，请求方式为get
@app.get("/")
def root():
    return {"message":"hello"}

@app.get("/users")
def get_users():
    return [{"zhang":"13","wang":"11"}]

#启动fastapi服务 uvicorn: python中的轻量级web服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
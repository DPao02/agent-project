from fastapi import APIRouter

api_cbs = APIRouter()

@api_cbs.get("/get")
async def read_get():
    return {"method": "get方法"}

@api_cbs.put("/put")
def read_put():
    return {"method": "put方法"}

@api_cbs.post("/post")
def read_post():
    return {"method": "post方法"}

@api_cbs.delete("/delete")
def read_delete():
    return {"method": "delete方法"}
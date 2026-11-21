"""显式创建MySQL业务表：不启动Agent、不请求模型、不写入演示订单。"""

from pathlib import Path

from dotenv import load_dotenv

from business.database import Database


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    database = Database.from_env()
    try:
        database.create_tables()
        database.check_ready()
        print("MySQL业务表已就绪：orders、order_items、tickets、ticket_events")
    finally:
        database.close()


if __name__ == "__main__":
    main()

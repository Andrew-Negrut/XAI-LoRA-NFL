from app.gradio_app import demo

if __name__ == "__main__":
    demo.queue(concurrency_count=2).launch()

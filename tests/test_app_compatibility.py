def test_root_app_reexports_gradio_client():
    import app
    from conductor_main import app as packaged_app

    assert app.create_demo is packaged_app.create_demo
    assert app.run_loop is packaged_app.run_loop

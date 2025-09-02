"""Consolidated configuration module"""
class Settings:
    def __init__(self):
        self.tesseract_cmd = 'tesseract'
        self.ocr_language = 'eng'

settings = Settings()

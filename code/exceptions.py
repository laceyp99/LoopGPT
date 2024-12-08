'''
This file contains the custom exceptions that are raised by the music generation API.
'''
class MusicGenerationError(Exception):
    pass

class ResponseValidationError(MusicGenerationError):
    pass

class AuthenticationError(MusicGenerationError):
    pass

class RateLimitError(MusicGenerationError):
    pass

class APICallError(MusicGenerationError):
    pass

class NetworkError(MusicGenerationError):
    pass
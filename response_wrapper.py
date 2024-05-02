class ResponseWrapper:
    def __init__(self, response):
        self.response = response

    def __getattr__(self, name):
        return getattr(self.response, name)

    def content(self):
        return self.response["choices"][0]["message"]["content"]

    def usage(self):
        usage = self.response["usage"]
        return {
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
        }

    def model(self):
        return self.response["model"]

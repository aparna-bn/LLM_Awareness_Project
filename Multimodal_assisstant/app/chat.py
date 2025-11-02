class ChatSession:
    def __init__(self):
        self.image_context = None
        self.image_question_answered = False

    def handle_upload(self, image):
        """Store uploaded image for one-time reference."""
        self.image_context = image
        self.image_question_answered = False

    def handle_question(self, question, process_func):
        """Route question based on image state."""
        if self.image_context and not self.image_question_answered:
            response = process_func(question, self.image_context.read())
            self.image_question_answered = True
        else:
            # All subsequent questions are general
            response = process_func(question)
        return response

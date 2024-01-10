import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk.chat.util import Chat, reflections
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Define conversation patterns
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!']),
    (r'how are you', ['I am good, thank you!', 'I am doing well.']),
    (r'what is your name', ['I am a chatbot.', 'You can call me Chat.']),
    (r'bye|goodbye', ['Goodbye!', 'See you later.']),
    (r'your favorite (.*)', ["I don't haave preferences as I'm just a program."]),
    (r'I like (.*)', ["That\'s great! What do you like about it?"]),
    (r'I need (.*)', ['Why do you need {0}?']),
    (r'(.*) (weather|temperature) (.*)', ['I am not equipped to provide real-time weather information.']),
    (r'what can you do', ['I can chat with you, provide information, and answer questions.']),
    (r'who are you', ['I am a chatbot using NLTK and Python.']),
    (r'how can you help', ['I can assist you with information, answer questions, and have casual conversations.']),
    (r'what is the meaning of life', ['The meaning of life is a philosophical  question with no definitive answer.']),
    (r'(.*) (hobby|hobbies)', ['I enjoy chatting and helping users. What about you?']),
    (r'(.*) (movie|movies)', ["I enjoy all kinds of movies. What\'s your favorite genre?"]),
    (r'(.*) (book|books)', ['I love reading! Do  you have a favorite book?']),
    (r'(.*) (music|song|songs)', ["I like various types of music. What\'s your favorite genre?"]),
    (r'(.*)', ["I didn\'t quite understand that. Could you please rephrase or ask something else?"]),
]

# Create the chatbot
chatbot = Chat(patterns, reflections)

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Memory
memory = {'name': None, 'feelings': None}

# Interaction loop
print("ChatBot: Hi! I'm your chatbot. Type 'exit' to end.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("ChatBot: Goodbye!")
        break
    
    # Perform sentiment analysis
    sentiment_score = sia.polarity_scores(user_input)['compound']
    if sentiment_score >= 0.05:
        memory['feelings'] = 'positive'
    elif sentiment_score <= 0.05:
        memory['feelings'] = 'negative'
    else:
        memory['feelings'] = 'neutral'
    
    # Perform part-of-speech tagging
    words = word_tokenize(user_input)
    pos_tags = pos_tag(words)
    
    # Convert tags to a simplified format for easier pattern matching
    simplified_tags = [(word, nltk.map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tags]
    
    # Concatenate words to form a sentence
    user_sentence = ' '.join(word for word, tag in simplified_tags)
    
    response = chatbot.respond(user_sentence)
    
    # Update memory with user's name
    if memory['name'] is None:
        name_pattern = r'my name is (.*)'
        match = nltk.regexp_tokenize(user_input, name_pattern)
        if match:
            memory['name'] = match[0]
            
    # Dynamic responses based on memory
    if 'feelings' in memory and memory['feelings'] == 'positive':
        print("ChatBot:", "That's wonderful to hear!")
    elif 'feelings' in memory and memory['feelings'] == 'negative':
        print("ChatBot:", "I'm sorry to hear that. Is there something specific on your mind?")
    else:
        print("ChatBot:", response)
    
    print("ChatBot:", response)
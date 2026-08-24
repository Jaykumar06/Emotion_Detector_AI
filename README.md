# 🧠 Emotion Detection & Message Analysis

An AI-powered **Emotion Detection and Message Analysis** application built with **Python, Streamlit, LangChain, and Groq**.

The application analyzes user-provided messages and uses a Large Language Model (LLM) to identify the emotional tone and provide an intelligent interpretation of the message.

## 🚀 Features

* 💬 Analyze text messages using AI
* 😊 Detect emotions from user input
* 🧠 LLM-powered message understanding
* ⚡ Fast responses using Groq
* 🔗 LangChain integration
* 🖥️ Interactive Streamlit interface
* 🔐 Environment-variable support for API keys
* 📦 Easy installation using `requirements.txt`

## 🛠️ Technologies Used

| Technology    | Purpose                             |
| ------------- | ----------------------------------- |
| Python        | Core programming language           |
| Streamlit     | Web application interface           |
| LangChain     | LLM application framework           |
| Groq          | Fast LLM inference                  |
| python-dotenv | Environment variable management     |
| Git & GitHub  | Version control and project hosting |

## 📂 Project Structure

```text
Emotion-Detection/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR-REPOSITORY
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages using:

```bash
python -m pip install -r requirements.txt
```

## 🔑 API Key Configuration

This project requires a **Groq API key**.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never upload your `.env` file or expose your API key on GitHub.

Add `.env` to your `.gitignore` file:

```text
.env
.venv/
__pycache__/
```

## ▶️ Run the Application

After installing the dependencies and configuring your API key, run:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL where you can open the application in your browser.

Usually it will be:

```text
http://localhost:8501
```

## 🧩 How It Works

The basic workflow of the application is:

```text
User enters a message
        ↓
Streamlit Interface
        ↓
LangChain Processing
        ↓
Groq LLM
        ↓
Emotion / Message Analysis
        ↓
Result displayed to the user
```

## 💡 Example

### Input

```text
I am really excited about my new project!
```

### Possible Output

```text
Emotion: Excitement / Happiness

The message expresses positive emotions, particularly
excitement and enthusiasm about starting a new project.
```

## 🎯 Use Cases

This project can be useful for:

* 💬 Social media sentiment and emotion analysis
* 📱 Message analysis
* 🤖 AI-powered chat applications
* 📊 Customer feedback analysis
* 🎓 Learning and demonstrating LLM integration
* 🧠 Natural Language Processing projects

## 🔮 Future Improvements

Some possible improvements include:

* Add more detailed emotion categories
* Add sentiment scores
* Support multiple languages
* Add conversation history
* Add emotion visualization using charts
* Store analysis history in a database
* Add user authentication
* Deploy the application online
* Add voice input and speech analysis

## 📋 Requirements

Make sure you have:

* Python 3.9+
* A Groq API key
* Internet connection
* Required Python packages from `requirements.txt`

## 🔒 Security

Never commit sensitive information such as:

```text
.env
API keys
passwords
secret tokens
```

Use environment variables instead.

## 👨‍💻 Author

**Jay Kumar**

Computer Science Student

### ⭐ Support

If you find this project useful, consider giving the repository a **⭐ Star** on GitHub!

---

## 📄 License

This project is created for educational and learning purposes.

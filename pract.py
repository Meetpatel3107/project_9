
import streamlit as st
import requests
import json
from groq import Groq

import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


client = Groq(api_key=GROQ_API_KEY)

def get_news_articles(query, max_articles=5):
    """Fetch news articles from NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': NEWSAPI_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': max_articles
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        articles = response.json().get('articles', [])
        return articles
    
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def summarize_with_groq(query, articles):
    """Summarize articles using Groq API"""
    try:
        # Prepare article text
        article_texts = []
        for i, article in enumerate(articles[:5], 1):
            title = article.get('title', 'No title')
            description = article.get('description', 'No description')
            article_texts.append(f"{i}. {title}\n{description}")
        
        articles_text = "\n\n".join(article_texts)
        
        # Create prompt
        prompt = f"""
        You are an AI assistant helping with equity research. 
        
        Query: {query}
        
        News Articles:
        {articles_text}
        
        Please provide a comprehensive summary of these articles in relation to the query. 
        Focus on key insights, trends, and important information for equity research.
        """
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",  
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error with Groq API: {str(e)}")
        return "Error generating summary"

def main():
    st.title("📰 News Research Tool")
    st.write("Enter your query to get the latest news articles summarized using Groq AI")
    
    # Sidebar for configuration
    st.sidebar.header("Settings")
    max_articles = st.sidebar.slider("Max articles to fetch", 3, 10, 5)
    
    # Main interface
    query = st.text_input("Enter your research query:", placeholder="e.g., Tesla stock performance, Apple earnings")
    
    if st.button("🔍 Get News Summary", type="primary"):
        if query:
            with st.spinner("Fetching news articles..."):
                articles = get_news_articles(query, max_articles)
            
            if articles:
                st.success(f"Found {len(articles)} articles")
                
                with st.spinner("Generating AI summary..."):
                    summary = summarize_with_groq(query, articles)
                
                # Display results
                st.subheader("📋 AI Summary")
                st.write(summary)
                
                # Show source articles
                with st.expander("📖 Source Articles"):
                    for i, article in enumerate(articles, 1):
                        st.write(f"**{i}. {article.get('title', 'No title')}**")
                        st.write(f"Source: {article.get('source', {}).get('name', 'Unknown')}")
                        st.write(f"Description: {article.get('description', 'No description')}")
                        if article.get('url'):
                            st.write(f"[Read full article]({article['url']})")
                        st.write("---")
            else:
                st.warning("No articles found for your query. Try a different search term.")
        else:
            st.warning("Please enter a query to search for news.")

if __name__ == "__main__":
    main()
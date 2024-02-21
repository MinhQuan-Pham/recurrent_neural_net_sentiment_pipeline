import requests
from bs4 import BeautifulSoup
import mysql.connector

# Function to create a connection and a table in Google Cloud SQL
def create_database():
    conn = mysql.connector.connect(
        user='your_username',
        password='your_password',
        host='your_google_cloud_sql_ip',
        database='your_database_name'
    )
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INT AUTO_INCREMENT PRIMARY KEY,
            company_name VARCHAR(255),
            review_text TEXT
        )
    ''')

    conn.commit()
    return conn, cursor

# Function to insert a review into the Google Cloud SQL database
def insert_review(conn, cursor, company_name, review_text):
    cursor.execute(f'''
        INSERT INTO reviews (company_name, review_text)
        VALUES ({company_name}, {review_text})
    ''')

    conn.commit()

# Function to scrape reviews and store them in the Google Cloud SQL database
def scrape_and_store_reviews(url, company_name):
    conn, cursor = create_database()

    # Make a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Locate the HTML elements containing the reviews
        review_elements = soup.find_all('div', class_='customer-review') 
        
        # Extract and store the reviews in the Google Cloud SQL database
        for review in review_elements:
            review_text = review.find('p', class_='review-text').text
            insert_review(conn, cursor, company_name, review_text)
            print(f"Review inserted into the Google Cloud SQL database: {review_text}")
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

    conn.close()

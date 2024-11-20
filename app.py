from flask import Flask, request, render_template
from data_preprocessing import preprocess_text
from feature_extraction import extract_features
from matching_algorithm import match_jobs_to_resume
import requests
from bs4 import BeautifulSoup
from termcolor import colored
from tabulate import tabulate

app = Flask(__name__)

API_URL = "https://mock-job-api.com/v1/jobs"  
API_KEY = "YOUR_API_KEY"

def fetch_job_postings(location):
    url = f'https://job-matcher-kmuw.onrender.com/api/jobs/location/{location}'

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 403:
        print("Access Forbidden. 403 error.")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    job_listings = []

    for job in soup.find_all('li'):  
        job_title = None
        description = None
        location = None
        
        for p in job.find_all('p'):
            text = p.text.strip()
            if text.startswith("Job:"):
                job_title = text.split(":", 1)[1].strip()
            elif text.startswith("Description:"):
                description = text.split(":", 1)[1].strip()
            elif text.startswith("Location:"):
                location = text.split(":", 1)[1].strip().upper()

        job_listings.append({
            'jobTitle': job_title or 'No title',
            'jobDescription': description or 'No description',
            'location': location or 'No location'
        })

    return job_listings

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/process_resume', methods=['POST'])
def process_resume():
    resume_text = request.form['resume']
    cleaned_text = preprocess_text(resume_text)
    resume_features = extract_features(cleaned_text)    
    return render_template('results.html', resume_features=resume_features)



@app.route('/match_jobs', methods=['POST'])
def match_jobs():
    resume_text = request.form['resume']
    location = request.form['location']

    job_listings = fetch_job_postings(location)
    cleaned_resume = preprocess_text(resume_text)
    resume_features = extract_features(cleaned_resume)
    print(colored("\n[INFO] Matching resume features to job listings.", 'yellow'))
    matches = match_jobs_to_resume(resume_features, job_listings)
    print(colored("\n[INFO] Matching Job Results (Table Format):", 'magenta'))
    if matches:
        table_data = [
            [
                "Job " + str(i+1),
                match["job"]["jobTitle"],
                f"{match['score']:.2f}",
                match["job"]["jobDescription"][:50] + "...",  # Truncate long descriptions
                match["job"]["location"],
            ]
            for i, match in enumerate(matches)
        ]
        print(tabulate(table_data, headers=["Job ID", "Job Title", "Similarity Score", "Description", "Location"], tablefmt="fancy_grid"))
    else:
        print(colored("\nNo matching jobs found.", 'red'))

    return render_template('matches.html', matches=matches)

if __name__ == '__main__':
    app.run(debug=True)

import math
from collections import Counter
from bs4 import BeautifulSoup
from tabulate import tabulate
from termcolor import colored
from data_preprocessing import lemmitizeTokens

import re

def clean_job_description(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    lemmatized_tokens=lemmitizeTokens(tokens)
    return lemmatized_tokens

def clean_html(html):
    return BeautifulSoup(html, "html.parser").get_text()

def compute_tf(doc):
    tf = Counter(doc)
    doc_len = len(doc)
    print(colored(f"\nDocument Length: {doc_len}", 'blue'))
    for word in tf:
        tf[word] = tf[word] / doc_len
    print(colored(f"\nTerm Frequencies (TF):", 'green'))
    for word, frequency in tf.items():
        print(f"{colored('Term:', 'yellow')} {word}, {colored('Frequency:', 'cyan')} {frequency}")
    return tf

def compute_idf(corpus):
    idf = {}
    N = len(corpus)
    doc_count = Counter()
    print(colored(f"\nTotal Documents (N): {N}", 'blue'))
    for doc in corpus:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_count[term] += 1
    for term, count in doc_count.items():
        idf[term] = math.log(N / (1 + count)) + 1
        print(f"{colored('Term:', 'yellow')} {term}, {colored('Document Frequency:', 'cyan')} {count}, {colored('IDF:', 'magenta')} {idf[term]}")
    return idf

def compute_tfidf(corpus):
    idf = compute_idf(corpus)
    tfidf_docs = []
    for doc in corpus:
        tf = compute_tf(doc)
        tfidf = {term: tf.get(term, 0) * idf.get(term, 0) for term in doc}
        tfidf_docs.append(tfidf)
        print(colored(f"\nTF-IDF for Document:", 'yellow'))
        for term, score in tfidf.items():
            print(f"{colored('Term:', 'yellow')} {term}, {colored('TF-IDF:', 'cyan')} {score}")
    return tfidf_docs, idf

def cosine_similarity_manual(vec1, vec2):
    print(colored("\nComparing vectors:", 'blue'), vec1, vec2)
    
    # Calculate dot product
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in set(vec1) | set(vec2))
    print(colored(f"\nDot Product Calculation:", 'green'))
    for term in set(vec1) | set(vec2):
        print(f"{colored('Term:', 'yellow')} {term}, {colored('vec1:', 'cyan')} {vec1.get(term, 0)}, {colored('vec2:', 'magenta')} {vec2.get(term, 0)}, {colored('Product:', 'red')} {vec1.get(term, 0) * vec2.get(term, 0)}")
    print(f"{colored('Dot Product:', 'blue')} {dot_product}")

    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))
    print(colored(f"\nMagnitude Calculation:", 'green'))
    print(f"{colored('Magnitude 1 Calculation:', 'yellow')} sqrt({sum(val**2 for val in vec1.values())}) = {magnitude1}")
    print(f"{colored('Magnitude 2 Calculation:', 'yellow')} sqrt({sum(val**2 for val in vec2.values())}) = {magnitude2}")

    if magnitude1 == 0 or magnitude2 == 0:
        print(colored("One of the vectors has zero magnitude, returning similarity of 0.", 'red'))
        return 0
    
    # Final cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    print(colored(f"\nCosine Similarity Calculation:", 'magenta'))
    print(f"{colored('Cosine Similarity:', 'yellow')} {dot_product} / ({magnitude1} * {magnitude2}) = {similarity}")
    return similarity

def match_jobs_to_resume(resume_features, job_listings):
    print(colored("\nMatching jobs to resume features:", 'blue'), resume_features)
    
    resume_text = " ".join(resume_features["skills"]).split()
    
    # Directly iterate over job_listings if it's a list
    jobs = [clean_job_description(job["jobDescription"]) for job in job_listings]
    
    print(colored("\nJob descriptions cleaned for matching:", 'green'))
    print(tabulate([["Job " + str(i+1)] + job for i, job in enumerate(jobs)], headers=["Job", "Description"], tablefmt="fancy_grid"))
    
    all_texts = [resume_text] + jobs
    tfidf_docs, idf = compute_tfidf(all_texts)
    
    print(colored("\nTF-IDF for each document:", 'yellow'))
    for i, tfidf in enumerate(tfidf_docs):
        print(colored(f"Document {i+1}:", 'red'))
        print(tabulate(tfidf.items(), headers=["Term", "TF-IDF"], tablefmt="fancy_grid"))

    similarities = []
    for job_tfidf in tfidf_docs[1:]:
        sim = cosine_similarity_manual(tfidf_docs[0], job_tfidf)
        similarities.append(sim)
    
    print(colored("\nCosine similarity scores:", 'magenta'), similarities)
    
    matches = [{"job": job_listings[i], "score": similarities[i]} for i in range(len(job_listings)) if similarities[i] > 0.01]
    matches.sort(key=lambda x: x["score"], reverse=True)
    
    print(colored("\nFiltered and sorted matches:", 'cyan'))
    if matches:
        print(tabulate([["Job " + str(i+1), match["job"]["jobTitle"], match["score"]] for i, match in enumerate(matches)], headers=["Job", "Title", "Similarity"], tablefmt="fancy_grid"))
    else:
        print(colored("No matching jobs found.", 'red'))
    
    return matches

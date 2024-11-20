import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

skills_keywords = [
    # General Programming Languages
    "Python", "PYTHON", "python3", "PYTHON3", "Py", "PY",
    "Java", "JAVA", "Core Java", "CORE JAVA", "Java SE", "JAVA SE", "Java EE", "JAVA EE",
    "JavaScript", "JAVASCRIPT", "JS", "js", "ECMAScript", "ECMASCRIPT",
    "SQL", "sql", "MySQL", "MYSQL", "T-SQL", "T-SQL", "PL/SQL", "PL/SQL", "PostgreSQL", "POSTGRESQL", "SQLite", "SQLITE", "MS SQL Server", "MS SQL SERVER",
    "HTML", "html", "HTML5", "html5",
    "CSS", "css", "CSS3", "css3", "SCSS", "scss", "SASS", "sass", "Stylus", "stylus",
    "C++", "c++", "Cpp", "cpp", "C++11", "c++11", "C++17", "c++17",
    "C#", "c#", "C Sharp", "C SHARP", ".NET C#", ".NET c#",
    "PHP", "php", "PHP7", "php7", "Laravel", "laravel", "PHP Laravel", "php laravel",

    # Frameworks & Libraries
    "React", "react", "React.js", "react.js", "ReactJS", "reactjs",
    "Next.js", "next.js", "Redux", "redux",
    "Node.js", "node.js", "NodeJS", "nodejs", "Express.js", "express.js", "ExpressJS", "expressjs",
    "Vue.js", "vue.js", "VueJS", "vuejs", "Angular", "angular", "AngularJS", "angularjs",
    "Django", "django", "Flask", "flask", "FastAPI", "fastapi",
    "Spring", "spring", "Spring Boot", "spring boot",
    "Bootstrap", "bootstrap", "Tailwind", "tailwind", "Foundation", "foundation",
    "jQuery", "jquery", "AJAX", "ajax",

    # Database & Query
    "MongoDB", "mongodb", "PostgreSQL", "postgresql", "Oracle DB", "oracle db", "Firebase", "firebase",
    "Elasticsearch", "elasticsearch", "Redis", "redis", "MariaDB", "mariadb",

    # DevOps & Cloud
    "AWS", "aws", "Amazon Web Services", "amazon web services",
    "Google Cloud Platform", "google cloud platform", "GCP", "gcp",
    "Azure", "azure", "Docker", "docker", "Kubernetes", "kubernetes",
    "CI/CD", "ci/cd", "Jenkins", "jenkins", "Git", "git", "GitHub", "github", "GitLab", "gitlab", "Bitbucket", "bitbucket",

    # Testing
    "JUnit", "junit", "Jest", "jest", "Mocha", "mocha", "Chai", "chai",
    "Selenium", "selenium", "Cypress", "cypress", "Postman", "postman",

    # Miscellaneous
    "REST API", "rest api", "GraphQL", "graphql",
    "JSON", "json", "XML", "xml",
    "TypeScript", "typescript", "TS", "ts",
    "Webpack", "webpack", "Babel", "babel",
    "Svelte", "svelte", "SvelteJS", "sveltejs",
    "Three.js", "three.js", "D3.js", "d3.js",
    "TailwindCSS", "tailwindcss", "Material UI", "material ui", "MUI", "mui", "Ant Design", "ant design",
    "WebSocket", "websocket", "Socket.IO", "socket.io",
    "OAuth", "oauth", "JWT", "jwt",
    "Linux", "linux", "Bash", "bash", "Shell Scripting", "shell scripting",
    "Python Flask", "python flask", "Python Django", "python django",
    "Pandas", "pandas", "NumPy", "numpy", "Scikit-learn", "scikit-learn",
    "TensorFlow", "tensorflow", "Keras", "keras",
    "Matplotlib", "matplotlib", "Seaborn", "seaborn", "PyTorch", "pytorch",
    "OpenCV", "opencv", "NLTK", "nltk", "Spacy", "spacy"
]

experience_keywords = ["internship", "job", "position", "role", "work", "experience"]

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp(skill) for skill in skills_keywords]
matcher.add("SKILLS", patterns)

def extract_features(tokens):
    
    doc = nlp(" ".join(tokens))
    skills_matches = matcher(doc)
    skills = [doc[start:end].text for match_id, start, end in skills_matches]
    
    experience = [token for token in tokens if token in experience_keywords]
    
    return {"skills": list(set(skills)), "experience": list(set(experience))}

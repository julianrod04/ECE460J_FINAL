import os
import pickle
import face_recognition
import cv2
import numpy as np
import wikipedia
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')


def load_image(image_path):
    image = face_recognition.load_image_file(image_path)
    print(f"Loaded image dtype: {image.dtype}, shape: {image.shape}")
    return image

def get_face_encodings(image):
    face_locations = face_recognition.face_locations(image, model="hog")
    return face_recognition.face_encodings(image, face_locations)

def build_encoding_database(dataset_path, encoding_file='celebrity_encodings.pkl'):
    encoding_path = os.path.join(dataset_path, encoding_file)
    if os.path.exists(encoding_path):
        with open(encoding_path, 'rb') as f:
            print("✅ Loaded existing encodings.")
            return pickle.load(f)

    print("Encoding file not found. Creating it now...")
    encoding_db = {}
    for person in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding_db[person] = encodings[0]
                    break
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    with open(encoding_path, 'wb') as f:
        pickle.dump(encoding_db, f)
    print("✅ Encodings saved.")
    return encoding_db

def identify_celebrity(face_encoding, database):
    names = list(database.keys())
    encodings = list(database.values())
    encodings = [e for e in encodings if isinstance(e, np.ndarray) and e.shape == (128,)]

    if not encodings:
        print("❌ No valid encodings found in the database.")
        return "Unknown"

    distances = face_recognition.face_distance(encodings, face_encoding)
    if len(distances) == 0:
        print("❌ No distances computed.")
        return "Unknown"

    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.6:  # Add threshold for confidence
        return names[best_match_index]
    else:
        print(f"⚠️ Low confidence match ({distances[best_match_index]:.2f})")
        return "Unknown"

def get_celebrity_summary(name):
    try:
        name_for_search = name.replace("_", " ")
        print(f"🔍 Searching Wikipedia for: {name_for_search}")
        
        # Try direct page lookup first
        try:
            page = wikipedia.page(name_for_search, auto_suggest=False)
            return page.title, page.content, page.url
        except wikipedia.exceptions.PageError:
            pass  # Fall through to search method
        except wikipedia.exceptions.DisambiguationError as e:
            # Try to find the exact match in disambiguation options
            for option in e.options:
                if option.lower() == name_for_search.lower():
                    page = wikipedia.page(option)
                    return page.title, page.content, page.url

        # If direct lookup fails, use search with improved filtering
        results = wikipedia.search(name_for_search)
        filtered_results = [
            r for r in results 
            if not any(bad in r.lower() 
                     for bad in ["filmography", "discography", "list of", " incident"])
        ]

        # Prioritize exact match or shortest result (most likely main page)
        filtered_results.sort(key=lambda x: (x.lower() != name_for_search.lower(), len(x)))
        
        for result in filtered_results:
            try:
                page = wikipedia.page(result, auto_suggest=False)
                # Secondary check for unwanted pages
                if any(bad in page.title.lower() 
                     for bad in ["filmography", "discography", "list of"]):
                    continue
                return page.title, page.content, page.url
            except (wikipedia.exceptions.PageError, 
                    wikipedia.exceptions.DisambiguationError):
                continue

        return "", "", ""
    except Exception as e:
        print(f"Error in Wikipedia search: {e}")
        return "", "", ""



def summarize_bio(name, content):
    """Extract general biographical information based on content analysis"""
    bio = []
    
    # Extract occupation
    occupation_pattern = r"(?i)is an? (?:American|British|Canadian|Australian|[A-Za-z]+) ((?:[A-Za-z]+(?:, | and )?)+)"
    occupation_match = re.search(occupation_pattern, content[:1000])
    if occupation_match:
        bio.append(f"• Occupation: {occupation_match.group(1).strip()}")
    
    # Extract birth date
    birth_date_pattern = r"(?i)born (?:on )?([A-Za-z]+ \d+, \d{4})"
    birth_match = re.search(birth_date_pattern, content[:1000])
    if birth_match:
        bio.append(f"• Born: {birth_match.group(1)}")
    
    # Extract nationality
    nationality_pattern = r"(?i)is an? ([A-Za-z]+) (actor|actress|singer|rapper|producer|director|writer|athlete|politician)"
    nationality_match = re.search(nationality_pattern, content[:1000])
    if nationality_match:
        bio.append(f"• Nationality: {nationality_match.group(1)}")
    
    # Extract notable works/awards
    notable_works = []
    
    # Look for award mentions
    for award in ["Academy Award", "Oscar", "Grammy", "Emmy", "Golden Globe"]:
        if re.search(f"(?i){re.escape(award)}", content):
            notable_works.append(f"{award} winner/nominee")
    
    # Extract famous for
    famous_for_patterns = [
        r"(?i)known for (?:her|his|their) ((?:role|work) (?:in|on) [^\.]+)",
        r"(?i)famous for ([^\.]+)",
        r"(?i)best known for ([^\.]+)"
    ]
    
    for pattern in famous_for_patterns:
        famous_match = re.search(pattern, content)
        if famous_match:
            bio.append(f"• Famous for: {famous_match.group(1).strip()}")
            break
    
    if notable_works:
        bio.append(f"• Awards/Recognition: {', '.join(notable_works)}")
    
    return "\n".join(bio) if bio else f"• No key facts extracted for {name}."

def extract_controversy_keywords():
    """Return a comprehensive list of controversy-related keywords"""
    return [
        # General controversy terms
        "controversy", "controversial", "scandal", "criticism", "backlash",
        "outcry", "protest", "uproar", "outrage", "condemn", "criticized",
        "allegations", "alleged", "accused", "apologized", "apology",
        
        # Legal issues
        "legal issues", "lawsuit", "sued", "court", "trial", "case", "settlement",
        "conviction", "arrested", "charged", "indicted", "investigation",
        
        # Personal scandals
        "affair", "divorce", "cheating", "infidelity", "addiction", "substance abuse",
        "rehab", "bankruptcy", "assault", "domestic", "violence", "abuse",
        
        # Professional controversies
        "fired", "cancelled", "dropped", "dispute", "feud", "banned", "prohibited",
        "blacklisted", "boycott", "declined", "rejected",
        
        # Social media/public relations
        "deleted", "offensive", "inappropriate", "insensitive", "slammed",
        "backlash", "negative", "viral", "twitter", "instagram", "comment",
        
        # Ethical issues
        "ethics", "misconduct", "unethical", "inappropriate", "harassment",
        "discrimination", "racist", "sexist", "homophobic", "xenophobic",
        "misogynistic", "inappropriate behavior"
    ]

def refine_controversy_list(controversies):
    """
    Refines the controversy list by grouping related topics and removing redundancy.
    """
    grouped_controversies = {}
    for controversy in controversies:
        # Determine the main subject of the controversy using keywords
        subject = None
        for keyword in extract_controversy_keywords():
            if keyword in controversy.lower():
                subject = keyword.capitalize()
                break

        # Group controversies by subject
        if subject:
            if subject not in grouped_controversies:
                grouped_controversies[subject] = []
            grouped_controversies[subject].append(controversy)
        else:
            # If no clear subject, add to 'Miscellaneous'
            if "Miscellaneous" not in grouped_controversies:
                grouped_controversies["Miscellaneous"] = []
            grouped_controversies["Miscellaneous"].append(controversy)

    # Format the refined list
    refined_list = []
    for subject, entries in grouped_controversies.items():
        # Combine related entries into a single bullet point
        combined_entry = " ".join(entries[:3])  # Limit to 3 entries per subject
        refined_list.append(f"• {subject}: {combined_entry}")

    return "\n".join(refined_list)


def summarize_controversies(controversies):
    """
    Summarizes controversies into concise bullet points with unique subjects.
    Each bullet point describes the controversy briefly and avoids redundancy.
    """
    summarized_controversies = []
    seen_subjects = set()

    for controversy in controversies:
        # Extract a brief summary of the controversy
        sentences = sent_tokenize(controversy["content"])
        if not sentences:
            continue

        # Use the first sentence as a summary
        summary = sentences[0]
        summary = re.sub(r"\(.*?\)", "", summary)  # Remove parenthesis
        summary = re.sub(r"\s+", " ", summary).strip(" .-")  # Clean spaces

        # Check for duplicate or similar subjects
        subject_keywords = ["divorce", "criticism", "controversy", "scandal", "feud", "lawsuit"]
        subject = next((kw.capitalize() for kw in subject_keywords if kw in summary.lower()), "Miscellaneous")
        
        if subject not in seen_subjects:
            summarized_controversies.append(f"• {subject}: {summary}.")
            seen_subjects.add(subject)

    return "\n".join(summarized_controversies[:5])  # Limit to 5 key points


def extract_controversies_improved(title, full_content, wiki_url):
    """
    Improved function to extract controversies from Wikipedia content and summarize them.
    """
    controversies = []
    controversy_keywords = extract_controversy_keywords()

    try:
        response = requests.get(wiki_url)
        soup = BeautifulSoup(response.content, "html.parser")

        headings = soup.find_all(['h2', 'h3', 'h4'])
        for heading in headings:
            heading_text = heading.get_text().lower()
            if "contents" in heading_text or "references" in heading_text or "notes" in heading_text:
                continue

            if any(keyword in heading_text for keyword in controversy_keywords):
                section_title = heading.get_text().replace("[edit]", "").strip()
                content = []
                current = heading.next_sibling
                while current and not (hasattr(current, 'name') and current.name in ['h2', 'h3', 'h4']):
                    if hasattr(current, 'name') and current.name in ['p', 'li']:
                        content.append(current.get_text().strip())
                    current = current.next_sibling
                if content:
                    controversies.append({
                        "section": section_title,
                        "content": " ".join(content)
                    })

        # Extract controversy-related sentences from full content
        sentences = sent_tokenize(full_content)
        potential_controversy_sentences = []
        for sentence in sentences:
            if len(sentence.split()) < 10:  # Skip short sentences
                continue

            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in controversy_keywords):
                name_parts = title.lower().split()
                if any(part in sentence_lower for part in name_parts) or \
                   any(pronoun in sentence_lower.split() for pronoun in ["he", "she", "they", "his", "her", "their"]):
                    potential_controversy_sentences.append(sentence)

        if potential_controversy_sentences:
            current_group = [potential_controversy_sentences[0]]
            for i in range(1, len(potential_controversy_sentences)):
                current_sentence = potential_controversy_sentences[i]
                prev_sentence = potential_controversy_sentences[i - 1]
                if full_content.find(current_sentence) - full_content.find(prev_sentence) < 500:
                    current_group.append(current_sentence)
                else:
                    if len(current_group) >= 2:
                        controversies.append({
                            "section": "Controversy Mention",
                            "content": " ".join(current_group)
                        })
                    current_group = [current_sentence]

            if len(current_group) >= 2:
                controversies.append({
                    "section": "Controversy Mention",
                    "content": " ".join(current_group)
                })

    except Exception as e:
        controversies.append({
            "section": "Error",
            "content": f"Error extracting controversies: {str(e)}"
        })

    unique_controversies = list({c["content"]: c for c in controversies}.values())  # Remove duplicates
    return summarize_controversies(unique_controversies)



def shorten_controversy_text(text):
    """
    Tokenize the paragraph into sentences and pick the most relevant ones.
    Clean up unnecessary text like parenthesis and extra spaces.
    """
    sentences = sent_tokenize(text)
    relevant_sentences = []

    for sentence in sentences:
        # Check if sentence contains name/pronoun and controversy term
        lowered = sentence.lower()
        if any(word in lowered for word in ["smith", "he", "his", "will"]) and any(
            kw in lowered for kw in extract_controversy_keywords()
        ):
            bullet = re.sub(r"\(.*?\)", "", sentence)  # Remove parenthesis
            bullet = re.sub(r"\s+", " ", bullet).strip(" .-")  # Clean spaces
            relevant_sentences.append(f"• {bullet}.")

    # Return formatted bullet points or fallback to first few sentences
    if relevant_sentences:
        return "\n".join(relevant_sentences[:5])  # Limit to 5 key points
    else:
        fallback = re.sub(r"\s+", " ", sentences[0]).strip(" .-")
        return f"• {fallback}."



def celebrity_recognition_system(image_path, dataset_path):
    image = load_image(image_path)
    face_encodings = get_face_encodings(image)

    if not face_encodings:
        print("❌ No faces detected in the image.")
        return

    face_encoding = face_encodings[0]
    print(f"\n🧬 Test image encoding shape: {face_encoding.shape}")

    database = build_encoding_database(dataset_path)
    print(f"\n🧠 Comparing to {len(database)} known encodings...")

    celebrity_name = identify_celebrity(face_encoding, database)
    print(f"\n✅ Identified: {celebrity_name}")

    if celebrity_name == "Unknown":
        print("⚠️ Face not recognized in dataset. Skipping Wikipedia search.")
        return

    title, full_content, wiki_url = get_celebrity_summary(celebrity_name)
    
    if not title or not full_content:
        print("⚠️ Could not find Wikipedia information for this celebrity.")
        return

    print(f"\n📚 Found Wikipedia page: {title}")
    
    print("\n📌 Key Bio Facts:")
    print(summarize_bio(celebrity_name, full_content))

    print("\n⚠️ Controversies and Noteworthy Incidents:")
    controversies = extract_controversies_improved(title, full_content, wiki_url)
    print(controversies)


# === MAIN USAGE ===
if __name__ == "__main__":
    test_image = "test_dataset/SJ-test4.jpg"
    dataset_folder = "celebrity_dataset"
    celebrity_recognition_system(test_image, dataset_path=dataset_folder) 

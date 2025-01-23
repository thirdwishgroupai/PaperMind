import json
import arxiv
import time
from typing import List, Dict
import random

def load_iclr_dataset(file_path: str) -> List[Dict]:
    """Load and parse the local ICLR 2025 dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_arxiv_ids(papers: List[Dict]) -> List[str]:
    """Extract arxiv IDs by searching paper titles."""
    arxiv_ids = []
    client = arxiv.Client()
    
    for paper in papers:
        try:
            title = paper.get('title', '').strip()
            if not title:
                continue
                
            # Search for the paper by title
            search = arxiv.Search(
                query=f'ti:"{title}"',  # Search in title
                max_results=1
            )
            
            try:
                result = next(client.results(search))
                # Extract arxiv ID from the entry ID (URL)
                arxiv_id = result.entry_id.split('/')[-1]
                arxiv_ids.append(arxiv_id)
                print(f"Found arXiv ID: {arxiv_id} for paper: {title}")
                
                # Sleep to respect API rate limits
                time.sleep(1)
                
            except StopIteration:
                print(f"No exact match found for paper: {title}")
            except Exception as e:
                print(f"Error searching for paper: {title} - {str(e)}")
        
        except Exception as e:
            print(f"Error processing paper: {str(e)}")
    
    print(f"\nTotal arXiv IDs found: {len(arxiv_ids)}")
    if len(arxiv_ids) == 0:
        print("No arXiv IDs found in the dataset")
    
    return arxiv_ids

def fetch_arxiv_papers(arxiv_ids: List[str], sample_size: int) -> List[Dict]:
    """Fetch paper details from arxiv API."""
    # Ensure sample size is within bounds and not larger than available papers
    available_papers = len(arxiv_ids)
    if available_papers < 50:
        print(f"Warning: Only {available_papers} papers available")
        sample_size = min(available_papers, sample_size)
    else:
        sample_size = min(sample_size, min(250, available_papers))
    
    print(f"Sampling {sample_size} papers from {available_papers} available papers...")
    
    # Randomly sample arxiv IDs
    selected_ids = random.sample(arxiv_ids, sample_size)
    
    # Create arxiv API client
    client = arxiv.Client()
    
    # Fetch papers
    papers = []
    for i, arxiv_id in enumerate(selected_ids, 1):
        try:
            print(f"Fetching paper {i}/{sample_size}: {arxiv_id}")
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search))
            
            paper_info = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'arxiv_id': arxiv_id,
                'pdf_url': result.pdf_url,
                'published': result.published.isoformat()
            }
            papers.append(paper_info)
            
            # Sleep to respect API rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching paper {arxiv_id}: {str(e)}")
    
    return papers

def main():
    # Configuration
    input_file = "example_iclr.json"
    sample_size = 50  # Starting with a smaller sample size for testing
    
    # Load local ICLR dataset
    print("Loading ICLR dataset...")
    iclr_papers = load_iclr_dataset(input_file)
    print(f"Loaded {len(iclr_papers)} papers from ICLR dataset")
    
    # Extract arxiv IDs by searching titles
    print("\nSearching arXiv for paper titles...")
    arxiv_ids = extract_arxiv_ids(iclr_papers)
    
    if len(arxiv_ids) == 0:
        print("No arXiv papers found. Exiting.")
        return
    
    # Fetch papers from arxiv
    print(f"\nFetching {sample_size} papers from arXiv...")
    papers = fetch_arxiv_papers(arxiv_ids, sample_size)
    
    # Save results
    output_file = "iclr2025_sample.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully downloaded {len(papers)} papers to {output_file}")

if __name__ == "__main__":
    main() 
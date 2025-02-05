import json
from typing import List, Dict
import requests
import time
from transformers import AutoTokenizer
import logging
from datetime import datetime
import os
import base64
from PIL import Image
from io import BytesIO

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/paper_processor_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")

class PaperProcessor:
    def __init__(self, base_url: str = "http://20.119.83.80:8080"):
        self.base_url = base_url
        logging.info(f"Initializing PaperProcessor with base_url: {base_url}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            logging.info("Successfully loaded GPT-2 tokenizer")
        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            raise
        
    def load_papers(self, file_path: str) -> List[Dict]:
        """Load papers from the ICLR sample JSON file."""
        logging.info(f"Loading papers from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            logging.info(f"Successfully loaded {len(papers)} papers")
            return papers
        except Exception as e:
            logging.error(f"Failed to load papers from {file_path}: {str(e)}")
            raise
    
    def submit_batch_conversion_jobs(self, papers: List[Dict]) -> str:
        """Submit multiple papers for batch conversion and return batch job ID."""
        logging.info(f"Submitting batch conversion job for {len(papers)} papers")
        
        try:
            # Create PDF directory if it doesn't exist
            pdf_dir = './pdf'
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir)
            
            # Prepare files for batch submission
            files = []
            for paper in papers:
                paper_id = paper.get('arxiv_id', 'unknown')
                pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
                
                # Download PDF if not cached
                if not os.path.exists(pdf_path):
                    pdf_url = paper.get('pdf_url')
                    if not pdf_url:
                        logging.warning(f"No PDF URL found for paper {paper_id}, skipping")
                        continue
                    
                    logging.info(f"Downloading PDF from {pdf_url}")
                    pdf_response = requests.get(pdf_url)
                    if pdf_response.status_code != 200:
                        logging.error(f"Failed to download PDF for {paper_id}: {pdf_response.status_code}")
                        continue
                    
                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_response.content)
                    logging.info(f"Saved PDF to {pdf_path}")
                
                # Add file to batch
                files.append(
                    ('documents', (f'paper_{paper_id}.pdf', open(pdf_path, 'rb'), 'application/pdf'))
                )
            
            # Submit batch conversion job
            response = requests.post(
                f"{self.base_url}/batch-conversion-jobs",
                files=files,
                data={
                    'extract_tables_as_images': 'true',
                    'image_resolution_scale': '2'
                }
            )
            
            # Close all opened files
            for _, (_, file_obj, _) in files:
                file_obj.close()
            
            if response.status_code != 200:
                raise Exception(f"Batch job submission failed: {response.text}")
            
            result = response.json()
            if 'job_id' not in result:
                raise Exception(f"Invalid response format: {result}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error submitting batch job: {str(e)}")
            raise

    def check_batch_job_status(self, batch_job_id: str) -> Dict:
        """Check status of a batch conversion job and return results if complete."""
        max_retries = 60  # Maximum number of status checks
        retry_delay = 5   # Seconds between checks
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/batch-conversion-jobs/{batch_job_id}")
                result = response.json()
                
                if result['status'] == 'SUCCESS':
                    logging.info(f"Batch job {batch_job_id} completed successfully")
                    return result['results']
                elif result['status'] == 'FAILED':
                    error_msg = result.get('error', 'Unknown error')
                    logging.error(f"Batch job {batch_job_id} failed: {error_msg}")
                    raise Exception(f"Batch conversion failed: {error_msg}")
                
                logging.debug(f"Batch job {batch_job_id} still processing (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                
            except Exception as e:
                logging.error(f"Error checking status for batch job {batch_job_id}: {str(e)}")
                raise
        
        raise Exception(f"Batch job {batch_job_id} timed out after {max_retries} attempts")

    def parse_sections(self, markdown_text: str) -> List[Dict]:
        """Parse markdown text into sections based on headers."""
        sections = []
        current_section = {"title": "", "content": []}
        
        for line in markdown_text.split('\n'):
            if line.startswith('#'):
                if current_section["title"]:
                    current_section["content"] = '\n'.join(current_section["content"])
                    sections.append(current_section)
                current_section = {
                    "title": line.strip('# '),
                    "content": []
                }
            else:
                current_section["content"].append(line)
        
        if current_section["title"]:
            current_section["content"] = '\n'.join(current_section["content"])
            sections.append(current_section)
        
        return sections

    def parse_references(self, markdown_text: str) -> List[str]:
        """Extract references section from markdown."""
        references = []
        ref_section = False
        
        for line in markdown_text.split('\n'):
            if line.lower().startswith('# reference'):
                ref_section = True
                continue
            if ref_section and line.strip():
                references.append(line.strip())
        
        return references

    def save_image(self, base64_str: str, filename: str, paper_id: str, img_type: str) -> str:
        """Save base64 encoded image to file and return relative path."""
        try:
            # Create image directories if they don't exist
            img_dir = os.path.join('structured_data_by_docling', paper_id, img_type)
            os.makedirs(img_dir, exist_ok=True)
            
            # Create full path for image
            img_path = os.path.join(img_dir, filename)
            
            # Decode and save image
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            img.save(img_path)
            
            # Return relative path from paper directory
            return os.path.join(img_type, filename)
            
        except Exception as e:
            logging.error(f"Error saving image {filename}: {str(e)}")
            return ""

    def process_papers(self, input_file: str):
        """Process all papers in batch and save results individually."""
        logging.info(f"Starting batch processing with input={input_file}")
        
        try:
            # Create necessary directories
            pdf_dir = './pdf'
            output_dir = './structured_data_by_docling'
            for directory in [pdf_dir, output_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    logging.info(f"Created directory: {directory}")
            
            # Load papers
            papers = self.load_papers(input_file)
            logging.info(f"Loaded {len(papers)} papers for processing")
            
            # Submit batch job
            batch_result = self.submit_batch_conversion_jobs(papers)
            batch_job_id = batch_result['job_id']
            
            # Wait for batch processing to complete
            conversion_results = self.check_batch_job_status(batch_job_id)
            
            # Process results for each paper
            processed_count = 0
            failed_count = 0
            
            for paper, result in zip(papers, conversion_results):
                paper_id = paper.get('arxiv_id', 'unknown')
                
                try:
                    # Create paper directory
                    paper_dir = os.path.join('structured_data_by_docling', paper_id)
                    os.makedirs(paper_dir, exist_ok=True)
                    
                    # Process the paper with conversion result
                    processed_paper = {
                        'arxiv_id': paper_id,
                        'title': paper['title'],
                        'authors': paper['authors'],
                        'pdf_url': paper.get('pdf_url', ''),
                        'sections': self.parse_sections(result.get('markdown', '')),
                        'references': self.parse_references(result.get('markdown', '')),
                        'figures': [],
                        'tables': [],
                        'chunks': result.get('chunks', []),
                        'topics': result.get('topics', []),
                        'innovations': result.get('innovations', []),
                        'summary': result.get('summary', ''),
                        'relationships': result.get('relationships', {}),
                        'processed_date': datetime.now().isoformat(),
                        'docling_version': result.get('version', 'unknown')
                    }
                    
                    # Process images
                    for image in result.get('images', []):
                        # Generate a safe filename
                        original_filename = image.get('filename', '')
                        safe_filename = ''.join(c for c in original_filename if c.isalnum() or c in '._-') or 'image.png'
                        
                        # Save image and get relative path
                        img_type = 'tables' if image.get('type') == 'table' else 'figures'
                        saved_path = self.save_image(
                            image.get('image', ''),
                            safe_filename,
                            paper_id,
                            img_type
                        )
                        
                        image_data = {
                            'filename': safe_filename,
                            'saved_path': saved_path,
                            'type': image.get('type', 'figure'),
                            'caption': image.get('caption', ''),
                            'page': image.get('page', 0)
                        }
                        
                        if image.get('type') == 'table':
                            processed_paper['tables'].append(image_data)
                        else:
                            processed_paper['figures'].append(image_data)
                    
                    # Save paper results
                    output_file = os.path.join(paper_dir, 'paper.json')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_paper, f, indent=2, ensure_ascii=False)
                    
                    processed_count += 1
                    logging.info(f"Successfully processed paper {paper_id}")
                    
                except Exception as e:
                    failed_count += 1
                    logging.error(f"Failed to process paper {paper_id}: {str(e)}")
            
            # Log final statistics
            logging.info("\nProcessing Summary:")
            logging.info(f"Total papers: {len(papers)}")
            logging.info(f"Successfully processed: {processed_count}")
            logging.info(f"Failed: {failed_count}")
            
            return processed_count, 0, failed_count
            
        except Exception as e:
            logging.error(f"Error in batch processing: {str(e)}")
            raise

def main():
    # Set up logging
    setup_logging()
    
    try:
        # Configuration
        input_file = "iclr2025_sample.json"
        base_url = "http://20.119.83.80:8080"
        
        logging.info("Starting paper processing pipeline")
        
        # Initialize processor
        processor = PaperProcessor(base_url)
        
        # Process papers
        processed, skipped, failed = processor.process_papers(input_file)
        
        logging.info("\nPipeline completed successfully")
        logging.info(f"Processed: {processed}, Skipped: {skipped}, Failed: {failed}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
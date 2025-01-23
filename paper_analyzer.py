import json
import os
from typing import List, Dict
import logging
from datetime import datetime
from collections import defaultdict
import networkx as nx
from transformers import AutoTokenizer, T5ForConditionalGeneration

def setup_logging():
    """Set up logging configuration."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/paper_analyzer_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")

class PaperAnalyzer:
    def __init__(self):
        model_name = "google/flan-t5-base"  # Better model for our text tasks
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.chunk_size = 500  # Reduced chunk size
        self.overlap = 100
        
    def load_paper(self, paper_path: str) -> Dict:
        """Load a processed paper from JSON."""
        with open(paper_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """Divide text into overlapping chunks with metadata."""
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunks = []
        
        for i in range(0, total_tokens, chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text,
                'token_count': len(chunk_tokens),
                'start_idx': i,
                'end_idx': i + len(chunk_tokens)
            }
            chunks.append(chunk)
            
            logging.debug(f"Created chunk {len(chunks)}: {chunk['token_count']} tokens "
                         f"(position {chunk['start_idx']}-{chunk['end_idx']})")
        
        logging.info(f"Text chunking stats: {total_tokens} total tokens split into {len(chunks)} chunks "
                    f"(avg {total_tokens/len(chunks):.1f} tokens per chunk)")
        return chunks
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        input_text = "Extract the main topics from this text: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            temperature=0.7,
            do_sample=True
        )
        topics = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(", ")
        return [t.strip() for t in topics if t.strip()]
    
    def extract_innovations(self, text: str) -> List[str]:
        """Extract key innovations from text."""
        input_text = "List the key innovations and contributions from this text: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            do_sample=True
        )
        innovations = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")
        return [i.strip() for i in innovations if i.strip()]
    
    def generate_summary(self, text: str) -> str:
        """Generate a comprehensive summary."""
        input_text = "Provide a detailed summary of this text: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def analyze_relationships(self, papers: List[Dict]) -> Dict:
        """Analyze relationships between papers and authors."""
        relationships = {
            'coauthorship': defaultdict(list),
            'citations': defaultdict(list),
            'topic_similarity': defaultdict(float),
            'author_papers': defaultdict(list)
        }
        
        # Build author and citation networks
        author_graph = nx.Graph()
        citation_graph = nx.DiGraph()
        
        for paper in papers:
            paper_id = paper['arxiv_id']
            
            # Process authors
            authors = paper['authors']
            for author in authors:
                relationships['author_papers'][author].append(paper_id)
                
            # Process coauthorships
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    author_graph.add_edge(author1, author2)
                    relationships['coauthorship'][author1].append(author2)
                    relationships['coauthorship'][author2].append(author1)
            
            # Process citations (from references)
            for ref in paper.get('references', []):
                citation_graph.add_edge(paper_id, ref)
                relationships['citations'][paper_id].append(ref)
        
        return relationships
    
    def analyze_paper(self, paper_path: str) -> Dict:
        """Analyze a single paper."""
        paper = self.load_paper(paper_path)
        paper_id = paper['arxiv_id']
        logging.info(f"Analyzing paper: {paper_id}")
        
        # Combine relevant text for analysis
        text_parts = []
        
        # Add title with weight
        text_parts.append(f"Title: {paper['title']}\n")
        
        # Add abstract if available
        if 'abstract' in paper:
            text_parts.append(f"Abstract: {paper['abstract']}\n")
        
        # Add sections
        for section in paper.get('sections', []):
            text_parts.append(f"## {section['title']}\n{section['content']}\n")
        
        text = "\n".join(text_parts)
        
        # Chunk the text
        chunks = self.chunk_text(text)
        logging.info(f"Created {len(chunks)} chunks for paper {paper_id}")
        
        # Analyze each chunk and combine results
        all_topics = set()
        all_innovations = set()
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks, 1):
            logging.info(f"Processing chunk {i}/{len(chunks)} of paper {paper_id} "
                        f"({chunk['token_count']} tokens)")
            
            topics = self.extract_topics(chunk['text'])
            innovations = self.extract_innovations(chunk['text'])
            summary = self.generate_summary(chunk['text'])
            
            all_topics.update(topics)
            all_innovations.update(innovations)
            chunk_summaries.append({
                'summary': summary,
                'token_count': chunk['token_count'],
                'position': {'start': chunk['start_idx'], 'end': chunk['end_idx']}
            })
        
        # Create final summary
        summary_texts = [cs['summary'] for cs in chunk_summaries]
        final_summary = self.generate_summary("\n".join(summary_texts))
        
        # Update paper with analysis results
        analysis_results = {
            'arxiv_id': paper_id,
            'title': paper['title'],
            'topics': list(all_topics),
            'innovations': list(all_innovations),
            'summary': final_summary,
            'chunks': [{'text': c['text'], 'token_count': c['token_count']} for c in chunks],
            'chunk_summaries': chunk_summaries,
            'analysis_metadata': {
                'total_tokens': sum(c['token_count'] for c in chunks),
                'chunk_count': len(chunks),
                'average_chunk_size': sum(c['token_count'] for c in chunks) / len(chunks),
                'processing_date': datetime.now().isoformat()
            }
        }
        
        # Save analysis results
        output_path = os.path.join(os.path.dirname(paper_path), 'analysis.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Paper {paper_id} analysis complete:")
        logging.info(f"- Total tokens: {analysis_results['analysis_metadata']['total_tokens']}")
        logging.info(f"- Chunks: {analysis_results['analysis_metadata']['chunk_count']}")
        logging.info(f"- Avg chunk size: {analysis_results['analysis_metadata']['average_chunk_size']:.1f} tokens")
        logging.info(f"- Topics found: {len(analysis_results['topics'])}")
        logging.info(f"- Innovations identified: {len(analysis_results['innovations'])}")
        
        return analysis_results
    
    def analyze_all_papers(self, data_dir: str = './structured_data_by_docling'):
        """Analyze all papers in the directory."""
        papers = []
        for paper_dir in os.listdir(data_dir):
            paper_path = os.path.join(data_dir, paper_dir, 'paper.json')
            if os.path.exists(paper_path):
                try:
                    analysis = self.analyze_paper(paper_path)
                    papers.append(analysis)
                except Exception as e:
                    logging.error(f"Error analyzing paper {paper_dir}: {str(e)}")
        
        # Analyze relationships across all papers
        relationships = self.analyze_relationships(papers)
        
        # Save relationship analysis
        relationship_path = os.path.join(data_dir, 'relationships.json')
        with open(relationship_path, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved relationship analysis to {relationship_path}")
        return papers, relationships

def main():
    setup_logging()
    
    try:
        analyzer = PaperAnalyzer()
        papers, relationships = analyzer.analyze_all_papers()
        
        logging.info("\nAnalysis Summary:")
        logging.info(f"Total papers analyzed: {len(papers)}")
        logging.info(f"Total authors: {len(relationships['author_papers'])}")
        logging.info(f"Total coauthorship connections: {len(relationships['coauthorship'])}")
        logging.info(f"Total citations: {sum(len(citations) for citations in relationships['citations'].values())}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
# PaperMind

## 🧠 Academic Paper Intelligence System

A comprehensive pipeline for downloading, processing, and analyzing academic papers from arXiv, with a focus on ICLR 2025 submissions.

## Overview

This project consists of three main components:
1. Paper Downloader - Downloads papers from arXiv
2. Paper Processor - Converts PDFs to structured data
3. Paper Analyzer - Analyzes paper content and relationships

## Features

- **Paper Download**
  - Searches arXiv by paper title
  - Downloads PDFs with caching
  - Rate-limiting for API compliance

- **PDF Processing**
  - Converts PDFs to structured text
  - Extracts figures and tables
  - Maintains document structure
  - Caches processed results

- **Content Analysis**
  - Text chunking with configurable size
  - Topic extraction
  - Innovation detection
  - Summary generation
  - Relationship analysis (coauthorship, citations)

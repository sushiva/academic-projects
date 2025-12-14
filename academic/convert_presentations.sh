#!/bin/bash

echo "Converting presentations to PDF and PowerPoint..."

# COVID presentation
cd covid-xray-detection/docs
pandoc presentation_content.md -o COVID_Presentation.pdf --pdf-engine=xelatex 2>/dev/null
pandoc presentation_content.md -o COVID_Presentation.pptx 2>/dev/null
echo "COVID presentations created"

# Helmet presentation  
cd ../../safety-helmet-detection/docs
pandoc presentation_content.md -o Helmet_Detection_Presentation.pdf --pdf-engine=xelatex 2>/dev/null
pandoc presentation_content.md -o Helmet_Detection_Presentation.pptx 2>/dev/null
echo "Helmet presentations created"

echo "Done! Check the docs folders for PDF and PPTX files"

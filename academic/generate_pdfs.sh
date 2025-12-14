#!/bin/bash

echo "Generating PDF presentations..."
echo ""

# COVID presentation
echo "Converting COVID-19 X-Ray presentation..."
cd covid-xray-detection/docs
pandoc presentation_content.md \
  -o COVID19_XRay_Classification_Presentation.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in \
  --toc \
  2>&1 | tail -5

if [ -f "COVID19_XRay_Classification_Presentation.pdf" ]; then
    echo "✓ COVID presentation PDF created!"
    ls -lh COVID19_XRay_Classification_Presentation.pdf
else
    echo "✗ COVID PDF generation failed (pandoc may not be installed)"
fi

echo ""

# Helmet presentation  
echo "Converting Safety Helmet Detection presentation..."
cd ../../safety-helmet-detection/docs
pandoc presentation_content.md \
  -o Safety_Helmet_Detection_Presentation.pdf \
  --pdf-engine=pdflatex \
  -V geometry:margin=1in \
  --toc \
  2>&1 | tail -5

if [ -f "Safety_Helmet_Detection_Presentation.pdf" ]; then
    echo "✓ Helmet presentation PDF created!"
    ls -lh Safety_Helmet_Detection_Presentation.pdf
else
    echo "✗ Helmet PDF generation failed (pandoc may not be installed)"
fi

echo ""
echo "Done! Check the docs folders for PDF files."
echo ""
echo "Files created:"
echo "  1. covid-xray-detection/docs/COVID19_XRay_Classification_Presentation.pdf"
echo "  2. safety-helmet-detection/docs/Safety_Helmet_Detection_Presentation.pdf"

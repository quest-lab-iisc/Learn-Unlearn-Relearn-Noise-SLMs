import csv
from indictrans import Transliterator

def main():
    # Initialize the transliterator
    trn = Transliterator(source='hin', target='eng', build_lookup=True)

    # Input and output file names
    input_file = 'path_to_input_csv'
    output_file = 'path_to_output_csv'

    # Read from input CSV and write to output CSV
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['input', 'translation', 'transliteration']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        # Process each row
        for row in reader:
            hindi_text = row['translation']
            transliteration = trn.transform(hindi_text)
            print(transliteration)
            
            # Write the row with all three columns
            writer.writerow({
                'input': row['input'],
                'translation': row['translation'],
                'transliteration': transliteration
            })

            print(f"Processed: {hindi_text} -> {transliteration}")

    print(f"Transliteration complete. Output saved to '{output_file}'")

if __name__ == "__main__":
    main()
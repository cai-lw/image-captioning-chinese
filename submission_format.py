import sys

with open(sys.argv[1], encoding='utf8') as fin, open('submission.txt', 'w', encoding='utf8') as fout:
    for i, ln in enumerate(fin):
        print(9000 + i, ' '.join(c for c in ln.strip()), file=fout)

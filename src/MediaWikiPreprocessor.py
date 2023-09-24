from langchain.document_loaders import MWDumpLoader
from dataclasses import dataclass
import xml.etree.ElementTree as ET

@dataclass
class MediaWikiPreprocessor:
    path:str

    def load_mediawiki_data(path, output_path):
        def strip_namespace(element):
            if '}' in element.tag:
                element.tag = element.tag.split('}', 1)[1]
            for child in element:
                strip_namespace(child)
                
        with open(path) as base_xml:
            print("Cleaning XML...")
            tree = ET.parse(base_xml)
            base_root = tree.getroot()
            clean_root = ET.Element("mediawiki")
            # Add boilerplat XML stuff
            clean_root.set("xmlns","http://www.mediawiki.org/xml/export-0.11/")
            #clean_root.set("xmlns:xsi","http://www.w3.org/2001/XMLSchema-instance")
            #clean_root.set("xsi:schemaLocation","http://www.mediawiki.org/xml/export-0.11/ http://www.mediawiki.org/xml/export-0.11.xsd")
            clean_root.set("version","0.11")
            clean_root.set("xml:lang","en")

            namespaces = ET.Element("namespaces")
            ns0 = ET.Element("namespace")
            ns0.set("key","0")
            ns0.set("case","first-letter")
            namespaces.append(ns0)
            clean_root.append(namespaces)

            for page in base_root:
                is_ok = False
                for elem in page:
                    if(elem.tag == "{http://www.mediawiki.org/xml/export-0.11/}ns" and 
                       elem.text is not None and
                       elem.text == "0"):
                        is_ok = True
                if(is_ok):
                    rev = page.find("{http://www.mediawiki.org/xml/export-0.11/}revision")
                    rev.remove(rev.find("{http://www.mediawiki.org/xml/export-0.11/}origin"))
                    strip_namespace(page)
                    clean_root.append(page)

            clean_tree = ET.ElementTree(clean_root)
            clean_tree.write(output_path)
        
        print("Loading cleaned XML into dataloader...")
        return MWDumpLoader(
            file_path = output_path, 
            encoding="utf8",
            namespaces = 0, 
            skip_redirects = True,
            stop_on_error = False, 
            ).load()

    def make_corpus():
        print("do something here")

if __name__ == "__main__":
    path = "data/en_gensinimpact_pages_current.xml"
    output_path = "data/clean_geshin_en.xml"
    docs = MediaWikiPreprocessor.load_mediawiki_data(path, output_path)
    print(f"There are {len(docs)} in the mwiki dump")
    print("Here is one of the docs:")
    print(docs[4])
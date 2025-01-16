
//https://dev.to/usmanaziz/extract-images-from-pdf-documents-using-c-net-207n
// https://products.groupdocs.com/parser/
// https://docs.groupdocs.com/parser/net/system-requirements/
using GroupDocs.Parser;
using GroupDocs.Parser.Data;
using GroupDocs.Parser.Options;
using System.Drawing;
using System.Drawing.Imaging;
// @"D:\repos2\c#\extract_images\extract_images\data\files\00003934.pdf" -->  works --> out {0}.Jpeg
// @"D:\repos2\c#\extract_images\extract_images\data\files\test.odt" -->  works --> out odt_{0}.Jpeg
// @"D:\repos2\c#\extract_images\extract_images\data\files\test.pptx" -->  works --> out pptx_{0}.Jpeg
// @"D:\repos2\c#\extract_images\extract_images\data\files\test.ppt" -->  works --> out ppt_{0}.Jpeg office 95 -2003
// @"D:\repos2\c#\extract_images\extract_images\data\files\test.rtf" -->  works --> out rtf_{0}.Jpeg
// @"D:\repos2\c#\extract_images\extract_images\data\files\test_images.docx" -->  works --> out docx_{0}.Jpeg
// @"D:\repos2\c#\extract_images\extract_images\data\files\test_images.doc" -->  works --> out doc_{0}.Jpeg office 95 -2003
// @"D:\repos2\c#\extract_images\extract_images\data\files\test_images.htm" -->  works --> out htm_{0}.Jpeg
// @"D:\repos2\c#\extract_images\extract_images\data\files\test_rtf.html" -->  works --> out html_{0}.Jpeg

using (Parser parser = new Parser(@"D:\repos2\c#\extract_images\extract_images\data\files\test.ppt"))
{
    // Extract images
    IEnumerable<PageImageArea> images = parser.GetImages();
    // Check if image extraction is supported
    if (images == null)
    {
        Console.WriteLine("Images extraction isn't supported");
        return;
    }

    int counter = 1;
    // Iterate over images
    foreach (PageImageArea image in images)
    {
        // Save each image
        Image.FromStream(image.GetImageStream()).Save(string.Format("D:\\repos2\\c#\\extract_images\\extract_images\\data\\out\\ppt_{0}.Jpeg", counter++), System.Drawing.Imaging.ImageFormat.Jpeg);
    }
}

IEnumerable<FileType> supportedFileTypes = FileType.GetSupportedFileTypes();

foreach (FileType fileType in supportedFileTypes)
{
    Console.WriteLine(fileType);
}


/*
 Microsoft Word Document(.doc)
Word Document Template(.dot)
Microsoft Word Open XML Document(.docx)
Word Open XML Macro-Enabled Document(.docm)
Word Open XML Document Template(.dotx)
Word Open XML Macro-Enabled Document Template(.dotm)
Plain Text File(.txt)
OpenDocument Text Document(.odt)
OpenDocument Document Template(.ott)
Rich Text Format File(.rtf)
Portable Document Format File(.pdf)
Hypertext Markup Language File(.html)
Hypertext Markup Language File(.xhtml)
MIME HTML File(.mhtml)
Markdown Files(.md)
XML File(.xml)
Compiled HTML Help File(.chm)
Open eBook File(.epub)
FictionBook(.fb2)
AZW3(.azw3)
FictionBook(.mobi)
Excel Spreadsheet(.xls)
Microsoft Excel Template(.xlt)
Microsoft Excel Open XML Spreadsheet(.xlsx)
Excel Open XML Macro-Enabled Spreadsheet(.xlsm)
Excel Binary Spreadsheet(.xlsb)
Excel Open XML Spreadsheet Template(.xltx)
Excel Open XML Macro-Enabled Spreadsheet Template(.xltm)
OpenDocument Spreadsheet(.ods)
OpenDocument Spreadsheet Template(.ots)
Comma Separated Values File(.csv)
Excel Add-In File(.xla)
Excel Open XML Macro-Enabled Add-In(.xlam)
PowerPoint Presentation(.ppt)
PowerPoint Slide Show(.pps)
PowerPoint Template(.pot)
PowerPoint Open XML Presentation(.pptx)
PowerPoint Open XML Macro-Enabled Presentation(.pptm)
PowerPoint Open XML Presentation Template(.potx)
PowerPoint Open XML Macro-Enabled Presentation Template(.potm)
PowerPoint Open XML Slide Show(.ppsx)
PowerPoint Open XML Macro-Enabled Slide(.ppsm)
OpenDocument Presentation(.odp)
OpenDocument Presentation Template(.otp)
Outlook Personal Information Store File(.pst)
Outlook Offline Data File(.ost)
E-Mail Message(.eml)
Apple Mail Message(.emlx)
Outlook Mail Message(.msg)
OneNote Document(.one)
Zipped File(.zip)
RAR Archive File(.rar)
Tar Archive File(.tar)
Bzip2 Archive File(.bz2)
Gzip Archive File(.gz)
Bitmap Image File(.bmp)
Graphical Interchange Format File(.gif)
JPEG 2000 Core Image File(.jp2)
JPEG Image(.jpg)
JPEG Image(.jpeg)
Portable Network Graphic(.png)
Tagged Image File(.tif)
Tagged Image File Format(.tiff)
*/
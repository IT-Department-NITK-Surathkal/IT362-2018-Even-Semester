% LaTeX Template for National Institute of Technology
% Karnataka, Surathkal.
%
% Author: Adithya Bhat
% Email: <dth.bht@gmail.com>

\NeedsTeXFormat{LaTeX2e}

\LoadClass[12pt,a4paper]{article}
\ProvidesClass{nitk}

\RequirePackage{graphicx}
\RequirePackage[dvips]{geometry}
\RequirePackage{tikz}
\RequirePackage{setspace}

\usetikzlibrary{calc}

\newcommand*{\report}[1]{\gdef\@report{#1}%
}
\newcommand*{\rollno}[1]{\gdef\@rollno{#1}%
}
\newcommand*{\name}[1]{\gdef\@name{#1}%
}
\newcommand*{\guide}[1]{\gdef\@guide{#1}%
}
\newcommand*{\dept}[1]{\gdef\@dept{#1}%
}
\newcommand*{\place}[1]{\gdef\@place{#1}%
}
\newcommand*{\years}[1]{\gdef\@years{#1}%
}

\def\and{\\ \vspace{0.5em}} % redefine so it is compatible with other classes

\setstretch{1.2}

\renewcommand{\maketitle}{% Make our beautiful front page
    \thispagestyle{empty}%
    \begin{tikzpicture}[remember picture, overlay]%
        \draw[line width = 1pt] ($(current page.north west) + (2em,-2em)$) rectangle ($(current page.south east) + (-2em,2em)$);%
    \end{tikzpicture}%
    
    {\centering
      \vspace{-3em}%
      {\Large\textbf{\@report}}\\%
      \vspace{1em}%
      On\\%
      \vspace{1em}%
      {\Huge \textbf{\@title}}\\%
      \vfill
      {\LARGE \bfseries Submitted by}\\%
      \vspace{1em}%
      {\Large \emph{\textbf{\@author}}}\\%
      \vspace{1em}%
      \vfill
      {\Large Under the Guidance of}\\%
      \vspace{1em}%
      {\Large \textbf{\@guide}}\\%
      \vspace{1em}%
      {\Large \textbf{\@dept,}}\\%
      \vspace{1em}%
      {\Large \textbf{NITK, Surathkal}}\\%
      \vspace{1em}%
      \vfill
      {\Large \textbf{Date of Submission: \@date}}\\%
      \vspace{1em}%
      \vfill
      \begin{figure}[!ht]%
          \centering
          \includegraphics{nitk-logo.png}%
      \end{figure}%
      {\Large \bfseries \@dept}\\%
      \vspace{0.5em}%
      {\Large \bfseries National Institute of Technology Karnataka, Surathkal.}\\% Hard-coded NITK Name
      \vspace{0.5em}%
      {\Large \bfseries \@years}%
      
      \newpage
    }
}

\let\origabstract\abstract
\let\origendabstract\endabstract

\renewcommand{\abstract}[1]{%
\pagenumbering{roman}
\section*{Abstract}
\addcontentsline{toc}{section}{Abstract}
#1
\newpage
}

\newcommand{\makedeclaration}[1]{%
% \pagenumbering{roman}
\addcontentsline{toc}{section}{Declaration}
\section*{Declaration}
#1\\

\begin{flushleft}
    Place: \@place\\
    Date: \@date
    \begin{flushright}
        ----------------------------- \\
        Signature of Shamitha S Udupa
        
    \end{flushright}
    \begin{flushright}
        ----------------------------- \\
        Signature of Sanjana B
        
    \end{flushright}
    
\end{flushleft}
\newpage
}

\newcommand{\acknowledge}[1]{
    % \thispagestyle{roman}
    \addcontentsline{toc}{section}{Acknowledgement}
    \section*{Acknowledgement}
    #1
    \newpage
}

\newcommand{\certificate}[1]{
    \addcontentsline{toc}{section}{Certificate}
    \section*{Certificate}
    #1\\
    \begin{flushleft}
    Place: \@place\\
    Date: \@date
    \begin{flushright}
        ----------------------------- \\
        (Name and Signature of the guide)
	\end{flushright}
    \end{flushleft}
    \newpage
    
}

\let\oldtoc\tableofcontents
\renewcommand{\tableofcontents}{
    \thispagestyle{empty}
    \oldtoc
    \newpage
}

\let\oldlof\listoffigures
\renewcommand{\listoffigures}{
    \thispagestyle{empty}
    \oldlof
    \newpage
}

\let\oldlot\listoftables
\renewcommand{\listoftables}{
    \thispagestyle{empty}
    \oldlot
    \newpage
}

\newcommand{\references}[1]{
    \addcontentsline{toc}{section}{References}
    \begin{thebibliography}{1}
    #1
    \end{thebibliography}
    \newpage
}
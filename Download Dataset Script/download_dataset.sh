#!/bin/bash

clear

declare -a url

# Array which contains the url of each of 9 pages of the websiste for How I Met Your Mother TV show
urlArray[0]=https://transcripts.foreverdreaming.org/viewforum.php?f=177
urlArray[1]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=25
urlArray[2]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=50
urlArray[3]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=75
urlArray[4]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=100
urlArray[5]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=125
urlArray[6]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=150
urlArray[7]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=175
urlArray[8]=https://transcripts.foreverdreaming.org/viewforum.php?f=177\&start=200

# File where are stored links of each episode 
fileLink=fileLinkEpisodes.txt
tmpfile=tmp_file.txt

# Empty the link file
echo "" > fileLink

# Extract the list of url for each episode
for i in $(seq 0 8)
do
	url=${urlArray[$i]}
	echo downloading $url...
	
	wget -qO- $url > $tmpfile
	# select only the link of the episodes without title
	grep 'href=' $tmpfile | grep '. class="topictitle">' | sed 's/.*href=\(.*\) /\1/' | awk '!/Happy New Year/' | \
	sed 's/ class="topictitle">/ /g' | sed 's/<\/a><\/h3>//g' | sed 's/ -.*//g' | \
	sed 's/"\.\/viewtopic/https:\/\/transcripts\.foreverdreaming\.org\/viewtopic/g' | sed 's/\&amp;t/\&t/g' |
	sed 's/\&amp;sid=\(.*\)"//g' >> $fileLink
	# with title
	# grep 'href=' $tmpfile | grep '. class="topictitle">' | sed 's/.*href=\(.*\) /\1/' | awk '!/Happy New Year/' | \
	# sed 's/ class="topictitle">/ /g' | sed 's/<\/a><\/h3>/ /g' | sed 's/ - / /g' | sed 's/ -/ /g' > $file
done

#cat $fileLink
echo ""
echo "### CREATING DATASET ###"
echo ""

# Create a directory where store the scripts
mkdir Dataset

# Extracting the script
while read line; do
	IFS=' ' read -ra lineSplt <<< "$line"
	url=${lineSplt[0]}
	echo Episode ${lineSplt[1]} link: $url
	fileName="./Dataset/${lineSplt[1]}.txt"
	wget -qO- $url | sed 's/\&quot;/"/g; s/<strong>//g; s/<\/strong>//g; s/<em>//g; s/<\/em>//g;' | 
	grep '^<p>[^<]' | sed 's/<p>//g; s/<\/p>//g'> $fileName
done < $fileLink
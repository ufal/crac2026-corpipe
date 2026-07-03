#!/bin/sh

set -e

# Download the gold train and dev, and zero-predicted dev and test data.
for conf in \
  train:https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-gold-train.zip \
  minidev.gold:https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-gold-minidev.zip \
  minidev:https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-input_blind-minidev.zip \
  minitest:https://ufal.mff.cuni.cz/~mnovak/files/crac26/unc-input_blind-minitest.zip
do
  split=${conf%%:*}
  url=${conf#*:}

  mkdir download
  wget $url -O download/$split.zip
  unzip -d download download/$split.zip
  for f in download/*.conllu; do
    lang=$(basename $f)
    lang=${lang%%-*}
    mkdir -p $lang
    mv $f $lang/$lang-corefud-$split.conllu
  done
  rm download/$split.zip download/*.json
  rmdir download   # Fails if not empty
done

echo All done

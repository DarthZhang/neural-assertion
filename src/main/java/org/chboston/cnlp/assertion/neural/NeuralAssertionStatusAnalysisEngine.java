package org.chboston.cnlp.assertion.neural;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.ctakes.core.util.DocumentIDAnnotationUtil;
import org.apache.ctakes.neural.feature.TokensSequenceWithWindowExtractor;
import org.apache.ctakes.typesystem.type.textsem.EntityMention;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.uima.UIMAFramework;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.Level;
import org.apache.uima.util.Logger;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;

public abstract class NeuralAssertionStatusAnalysisEngine extends
    CleartkAnnotator<String> {

  Logger logger = UIMAFramework.getLogger(NeuralAssertionStatusAnalysisEngine.class);
  private final static int WINDOW_SIZE = 10;
  private TokensSequenceWithWindowExtractor seqExtractor = new TokensSequenceWithWindowExtractor(WINDOW_SIZE);
//  private ContinuousTextExtractor continuousText;
//  protected CleartkExtractor<IdentifiedAnnotation,BaseToken> tokenVectorContext;
//  final String vectorFile = "org/chboston/cnlp/assertion/neural/mimic_vectors.txt";
//
//  @Override
//  public void initialize(UimaContext context)
//      throws ResourceInitializationException {
//    super.initialize(context);
//    try {
//      this.continuousText = new ContinuousTextExtractor(vectorFile, OovStrategy.EMPTY_VECTOR);
//      this.tokenVectorContext = new CleartkExtractor<>(
//          BaseToken.class,
//          continuousText,
//          new PrecedingWithPadding(5, this.continuousText.getEmbeddingsDimensionality()),
//          new SumContext(new Bag(new Covered())),
//          new FollowingWithPadding(5, this.continuousText.getEmbeddingsDimensionality())
//          );
//    } catch (CleartkExtractorException e) {
//      System.err.println("cannot find file: "+ vectorFile);
//      e.printStackTrace();
//    }
//  }
  
  @Override
  public void process(JCas jCas) throws AnalysisEngineProcessException {
    String documentId = DocumentIDAnnotationUtil.getDocumentID(jCas);
    if(documentId != null){
      logger.log(Level.INFO, "Processing document " + documentId);
    }

    Collection<IdentifiedAnnotation> entities = JCasUtil.select(jCas, IdentifiedAnnotation.class);
    for (IdentifiedAnnotation identifiedAnnotation : entities)
    {
      if (!(identifiedAnnotation instanceof EntityMention || identifiedAnnotation instanceof EventMention))
      {
        continue;
      }
      IdentifiedAnnotation target = identifiedAnnotation;

      List<Feature> feats = new ArrayList<>();

//      feats.addAll(this.tokenVectorContext.extract(jCas, target));
      feats.addAll(this.seqExtractor.extract(jCas, target));
      Instance<String> instance = new Instance<>(feats);
      
      if(this.isTraining()){
        String outcome = this.getLabel(target);
        
        instance.setOutcome(outcome);
        this.dataWriter.write(instance);
      }else{
        String label = this.classifier.classify(feats);
        this.applyLabel(target, label);
      }
    }

  }

  public abstract String getLabel(IdentifiedAnnotation ent);
  public abstract void applyLabel(IdentifiedAnnotation ent, String label);
}

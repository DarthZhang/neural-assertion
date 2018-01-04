package org.chboston.cnlp.assertion.neural;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.ctakes.assertion.eval.AnnotationStatisticsCompact;
import org.apache.ctakes.assertion.eval.AssertionEvaluation;
import org.apache.ctakes.assertion.eval.AssertionEvaluation.ReferenceAnnotationsSystemAssertionClearer;
import org.apache.ctakes.assertion.eval.AssertionEvaluation.ReferenceIdentifiedAnnotationsSystemToGoldCopier;
import org.apache.ctakes.assertion.eval.XMIReader;
import org.apache.ctakes.neural.ScriptStringFeatureDataWriter;
import org.apache.ctakes.typesystem.type.textsem.EntityMention;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.uima.UIMAFramework;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.pipeline.JCasIterator;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.Level;
import org.apache.uima.util.Logger;
import org.cleartk.eval.Evaluation_ImplBase;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.jar.DefaultDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.jar.JarClassifierBuilder;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class NeuralAssertionEvaluation extends Evaluation_ImplBase<File, Map<String, AnnotationStatisticsCompact<String>>> {
  public static class Options {
    @Option(
        name = "--train-dir",
        usage = "specify the directory containing the XMI training files (for example, /NLP/Corpus/Relations/mipacq/xmi/train)",
        required = true)
    public String trainDirectory;
    
    @Option(
        name = "--test-dir",
        usage = "specify the directory containing the XMI testing files (for example, /NLP/Corpus/Relations/mipacq/xmi/test)",
        required = true)
    public File testDirectory;
    
    @Option(
        name = "--skip-train",
        usage = "skip training and just evaluate on the specified test set",
        required = false)
    public boolean skipTrain=false;
    
    @Option(
        name = "--skip-write",
        usage = "Skip writing the data and just train and evaluate",
        required = false)
    public boolean skipWrite=false;
    
    @Option(
        name = "--write-only",
        usage = "Skip training and testing and just write the data with the data writer (for running parameter sweeps on the server)",
        required = false)
    public boolean writeOnly=false;
    
    @Option(
        name = "--baseline",
        usage = "Run baseline individual systems rather than multi-task system",
        required = false)
    public boolean baseline=false;
    
    @Option(
        name = "--data-dir",
        usage = "Directory where training data should be written",
        required=false)
    public String dataDir = "target/models";
  }
  
  public NeuralAssertionEvaluation(File baseDirectory) {
    super(baseDirectory);
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    CmdLineParser parser = new CmdLineParser(options);
    parser.parseArgument(args);

    File neuralDir = new File(options.dataDir);
    File configDir = new File(neuralDir, options.baseline ? "singletask" : "multitask");
    NeuralAssertionEvaluation eval = new NeuralAssertionEvaluation(configDir);

    List<File> trainFiles = new ArrayList<>();
    if(options.skipTrain){
      eval.skipTrain = true;
    }else if(options.skipWrite){
      eval.skipWrite = true;
    }else{
      if (null != options.trainDirectory) {
        String[] dirs = options.trainDirectory.split("[;:]");
        for (String dir : dirs) {
          File trainDir = new File(dir);
          if (trainDir.listFiles()!=null) {
            for (File f : trainDir.listFiles()) {
              trainFiles.add(f);
            }
          }
        }
      }
    }

    eval.baseline = options.baseline;
    eval.writeOnly = options.writeOnly;
    
    List<File> testFiles = new ArrayList<>();
    testFiles = Arrays.asList(options.testDirectory.listFiles());

    // if this is a new run make sure the models directory is empty:
    if(!(options.skipTrain || options.skipWrite)){
      FileUtils.deleteDirectory(new File(configDir, "train_and_test"));
    }
    
    Map<String, AnnotationStatisticsCompact<String>> stats = eval.trainAndTest(trainFiles, testFiles);
    if(options.writeOnly){
      System.out.println("No evaluation performed due to writeOnly flag");
      return;
    }
    
    AssertionEvaluation.printScore(stats, configDir.getAbsolutePath());
    double f1_ave = (stats.get("polarity").f1("-1") +
        stats.get("uncertainty").f1("1") +
        stats.get("generic").f1("true") +
        stats.get("conditional").f1("true") +
        stats.get("historyOf").f1("1")) / 5.0;
    
    System.out.println(String.format("Polarity: %.3f",  stats.get("polarity").f1("-1")));
    System.out.println(String.format("Uncertainty: %.3f", stats.get("uncertainty").f1("1")));
    System.out.println(String.format("Generic: %.3f", stats.get("generic").f1("true")));
    System.out.println(String.format("Conditional: %.3f", stats.get("conditional").f1("true")));
    System.out.println(String.format("HistoryOf: %.3f", stats.get("historyOf").f1("1")));
    System.out.println(String.format("Macro-f: %.3f\n " , f1_ave));
    
//    for(Map.Entry<String, AnnotationStatisticsCompact<String>> stat : stats.entrySet()){
//      System.out.println(stat.getKey());
//      System.out.println(stat.getValue().f1());
//    }
  }

  public static final Logger logger = UIMAFramework.getLogger(NeuralAssertionEvaluation.class);
  public boolean skipWrite = false;
  public boolean skipTrain = false;
  public boolean baseline = false;
  public boolean writeOnly = false;
  private String[] atts= new String[] {"polarity", "uncertainty", "conditional", "generic", "historyOf", "subject"}; 
      
  @Override
  protected CollectionReader getCollectionReader(List<File> items)
      throws Exception {
    String[] paths = new String[items.size()];
    for (int i = 0; i < paths.length; ++i) {
      paths[i] = items.get(i).getPath();
    }
    return CollectionReaderFactory.createReader(
        XMIReader.class,
        TypeSystemDescriptionFactory.createTypeSystemDescriptionFromPath(),
        XMIReader.PARAM_FILES,
        paths);
  }

  @Override
  protected void train(CollectionReader collectionReader, File directory)
      throws Exception {
    if(this.skipTrain) return;
    
    if(!this.skipWrite){
      AggregateBuilder builder = new AggregateBuilder();

      if(baseline){
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralPolarityAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            new File(directory, atts[0]),
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/singletask/" + atts[0]
            ) );
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralUncertaintyAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            new File(directory, atts[1]),
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/singletask/" + atts[1]
            ) );
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralConditionalAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            new File(directory, atts[2]),
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/singletask/" + atts[2]
            ) );
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralGenericAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            new File(directory, atts[3]),
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/singletask/" + atts[3]
            ) );
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralHistoryOfAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            new File(directory, atts[4]),
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/singletask/" + atts[4]
            ) );
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralSubjectAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            new File(directory, atts[5]),
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/singletask/" + atts[5]
            ) );
      }else{
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralMultitaskAssertionStatusAnalysisEngine.class,
            CleartkAnnotator.PARAM_IS_TRAINING,
            true,
            DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
            ScriptStringFeatureDataWriter.class,
            DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
            directory,
            ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
            "scripts/keras/multitask/"
            ) );
      }
      // run the pipeline and write out the data
      SimplePipeline.runPipeline(collectionReader,  builder.createAggregateDescription());
    }
    if(this.writeOnly) return;
    
    // call the classifier builder to build a classifier and then package it into a jar
    if(this.baseline){
      for(int i = 0; i < atts.length; i++){
        String att = atts[i];
        JarClassifierBuilder.trainAndPackage(new File(directory, att));
      }
    }else{
      JarClassifierBuilder.trainAndPackage(directory);
    }
  }

  @Override
  protected Map<String, AnnotationStatisticsCompact<String>> test(
      CollectionReader collectionReader, File directory) throws Exception {
    Map<String,AnnotationStatisticsCompact<String>> stats = new HashMap<>();
    if(this.writeOnly) return stats;
    AnnotationStatisticsCompact<String> polarityStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> conditionalStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> uncertaintyStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> subjectStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> genericStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> historyStats = new AnnotationStatisticsCompact<>();

    AggregateBuilder builder = new AggregateBuilder();

    // copy annotations to a gold view so we can evaluate 
    AnalysisEngineDescription goldCopierIdentifiedAnnotsAnnotator = AnalysisEngineFactory.createEngineDescription(ReferenceIdentifiedAnnotationsSystemToGoldCopier.class);
    builder.add(goldCopierIdentifiedAnnotsAnnotator);
    
    // remove gold from default view so we don't accidentally peak
    AnalysisEngineDescription assertionAttributeClearerAnnotator = AnalysisEngineFactory.createEngineDescription(ReferenceAnnotationsSystemAssertionClearer.class);
    builder.add(assertionAttributeClearerAnnotator);
    
    if(this.baseline){
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralPolarityAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(new File(directory, atts[0]), "model.jar").getPath()));
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralUncertaintyAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(new File(directory, atts[1]), "model.jar").getPath()));
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralConditionalAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(new File(directory, atts[2]), "model.jar").getPath()));
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralGenericAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(new File(directory, atts[3]), "model.jar").getPath()));
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralHistoryOfAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(new File(directory, atts[4]), "model.jar").getPath()));
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralSubjectAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(new File(directory, atts[5]), "model.jar").getPath()));
    }else{
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralMultitaskAssertionStatusAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(directory, "model.jar").getPath()));
    }
    JCasIterator casIter = new JCasIterator(collectionReader, builder.createAggregate());
    for ( ; casIter.hasNext();) {
      JCas jCas = casIter.next();
      JCas goldView;
      try {
        goldView = jCas.getView(AssertionEvaluation.GOLD_VIEW_NAME);
      } catch (CASException e) {
        logger.log(Level.INFO, "jCas.getViewName() = " + jCas.getViewName());
        throw new AnalysisEngineProcessException(e);
      }

      Collection<IdentifiedAnnotation> goldEntitiesAndEvents = getEntitiesAndEvents(goldView);
      Collection<IdentifiedAnnotation> systemEntitiesAndEvents = getEntitiesAndEvents(jCas);
      
      addStats(polarityStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "polarity");
      addStats(conditionalStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "conditional");
      addStats(uncertaintyStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "uncertainty");
      addStats(subjectStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "subject");
      addStats(genericStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "generic");
      addStats(historyStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "historyOf");
    }
    casIter.collectionProcessComplete();
    
    stats.put("polarity",  polarityStats);
    stats.put("conditional",  conditionalStats);
    stats.put("uncertainty",  uncertaintyStats);
    stats.put("subject", subjectStats);
    stats.put("generic", genericStats);
    stats.put("historyOf", historyStats);

    return stats;
  }

  private static void addStats(AnnotationStatisticsCompact<String> stats, Collection<IdentifiedAnnotation> goldSpans, Collection<IdentifiedAnnotation> systemSpans, String key){
    stats.add(goldSpans, systemSpans,
        AnnotationStatisticsCompact.<IdentifiedAnnotation>annotationToSpan(),
        AnnotationStatisticsCompact.<IdentifiedAnnotation>annotationToFeatureValue(key));
  }
  private static Collection<IdentifiedAnnotation> getEntitiesAndEvents(JCas jCas){
    Collection<IdentifiedAnnotation> entitiesAndEvents = new ArrayList<>();
    Collection<EntityMention> entities = JCasUtil.select(jCas, EntityMention.class);
    entitiesAndEvents.addAll(entities);
    Collection<EventMention> events = JCasUtil.select(jCas, EventMention.class);
    entitiesAndEvents.addAll(events);
    return entitiesAndEvents;
  }
}

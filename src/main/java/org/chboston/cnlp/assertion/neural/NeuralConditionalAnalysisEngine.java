package org.chboston.cnlp.assertion.neural;

import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;

public class NeuralConditionalAnalysisEngine extends
    NeuralAssertionStatusAnalysisEngine {

  @Override
  public String getLabel(IdentifiedAnnotation ent) {
    return String.valueOf(ent.getConditional());
  }

  @Override
  public void applyLabel(IdentifiedAnnotation ent, String label) {
    ent.setConditional(Boolean.parseBoolean(label));
  }

}

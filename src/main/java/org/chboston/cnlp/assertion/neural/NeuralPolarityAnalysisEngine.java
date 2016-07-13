package org.chboston.cnlp.assertion.neural;

import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;

public class NeuralPolarityAnalysisEngine extends
    NeuralAssertionStatusAnalysisEngine {

  @Override
  public String getLabel(IdentifiedAnnotation ent) {
    return String.valueOf(ent.getPolarity());
  }

  @Override
  public void applyLabel(IdentifiedAnnotation ent, String label) {
    ent.setPolarity(Integer.parseInt(label));
  }

}

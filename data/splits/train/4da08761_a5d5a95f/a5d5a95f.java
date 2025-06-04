import java.util.*;
import java.io.*;
public class Main {
// For fast input output

// end of fast i/o code
public static void main(String[] args) {
  FastReader reader = new FastReader();
  int Q = reader.nextInt();
  outer: for (int q = 0; q < Q; q++) {
    int N = reader.nextInt();
    int[][] scores = new int[5][N];
    for (int i = 0; i < N; i++) {
      int[] occurs = new int[5];
      String word = reader.next();
      for (int j = 0; j < word.length(); j++) {
        occurs[word.charAt(j) - 'a']++;
      }
      for (int j = 0; j < 5; j++) {
        scores[j][i] = occurs[j] - (word.length() - occurs[j]) ;
      }
    }
    int bestCount = 0;
    for (int i = 0; i < 5; i++) {
      int[] curr = scores[i];
      Arrays.sort(curr);
      int currentCount = 1;
      int currentScore = curr[curr.length - 1];
      for (int j = curr.length - 2; j >= 0 && currentScore > 0; j--) {
        currentScore += curr[j];
        currentCount++;
      }
      if (currentScore <= 0) currentCount--;
      bestCount = Math.max(currentCount, bestCount);
    }
    System.out.println(bestCount);
  }
}

}

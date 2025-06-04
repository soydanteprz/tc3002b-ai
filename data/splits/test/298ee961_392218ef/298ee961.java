import java.util.*;
import java.io.*;

public class Main {
  
  static void printLine(String s) {
    System.out.println(s);
  }
  
  static int parseInt(String s) {
    return Integer.parseInt(s.trim());
  }
  static String[] split(String s) {
    return s.split("\\s+");
  }
  
  public static void main(String[] args) throws Exception {
    BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    String line;
    StringTokenizer st;
    int test = parseInt(br.readLine());
    for (int t = 0; t < test; t++) {
      int n = parseInt(br.readLine());
      ArrayList<Integer> blue = new ArrayList<Integer>();
      ArrayList<Integer> red = new ArrayList<Integer>();
      String[] toks = split(br.readLine());
      String color = br.readLine();
      for (int i = 0; i < n; i++) {
        if (color.charAt(i) == 'B') {
          blue.add(parseInt(toks[i]));
        } else {
          red.add(parseInt(toks[i]));
        }
      }
      int[] blueArr = new int[blue.size()];
      int[] redArr = new int[red.size()];
      for (int i = 0; i < blue.size(); i++) {
        blueArr[i] = blue.get(i);
      }
      for (int i = 0; i < red.size(); i++) {
        redArr[i] = red.get(i);
      }
      Arrays.sort(blueArr);
      Arrays.sort(redArr);
      int blueP = 0; // pointer
      int redP = 0; // pointer
      
      // blue -> decrease, red -> increase
      int start = 1;
      boolean unable = false;
      while (start <= n) {
        if (blueP < blueArr.length) {
          if (blueArr[blueP] < start) {
            unable = true;
            printLine("NO");
            break;  
          } else {
            blueP++;
            start++;
          }
        } else {
          if (redArr[redP] > start) {
            unable = true;
            printLine("NO");
            break;  
          } else {
            redP++;
            start++;
          }
        }        
      }
      if (!unable) {
        printLine("YES");
      }      
    }
  }
}

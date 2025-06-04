import java.io.*;
import java.util.*;

public class E {
  public static void main(String[] args) throws IOException {
    FastReader fr = new FastReader();
    PrintWriter pr = new PrintWriter(new OutputStreamWriter(System.out));
    int t = fr.nextInt();

    while (t-- > 0) {
      int n = fr.nextInt(), k = fr.nextInt();
      String[] arr = fr.nextLine().split(" ");
      String[] temp = fr.nextLine().split(" ");

      long[] L = new long[n];
      long[] R = new long[n];

      long[] c = new long[n];
      Arrays.fill(c, Long.MAX_VALUE);

      for (int i = 0; i < k; i++) {
        c[toInt(arr[i]) - 1] = toInt(temp[i]);
      }

      long p = Integer.MAX_VALUE;

      for (int i = 0; i < R.length; i++) {
        p = Math.min(p + 1, c[i]);
        L[i] = p;
      }

      p = Integer.MAX_VALUE;

      for (int i = n - 1; i >= 0; i--) {
        p = Math.min(p + 1, c[i]);
        R[i] = p;
      }

      StringBuilder sb = new StringBuilder();

      for (int i = 0; i < R.length; i++) {
        sb.append(Math.min(R[i], L[i]) + " ");
      }

      pr.println(sb.toString().trim());
    }

    pr.close();
  }

  

  static int toInt(String s) {
    return Integer.parseInt(s);
  }

  
}

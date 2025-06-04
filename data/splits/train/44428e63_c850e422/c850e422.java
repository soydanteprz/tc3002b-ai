import java.io.*;
import java.util.*;

public class Code {

  public static void main(String[] args) {
    FastReader in = new FastReader();
    PrintWriter out = new PrintWriter(System.out);

    int t = in.nextInt();
    while (t-- > 0) {
      int n = in.nextInt();
      long a[] = new long[n];
      for (int i = 0; i < n; i++) {
        a[i] = in.nextLong();
      }
      long result = Long.MAX_VALUE;
      for (int i = 0; i < 2; i++) {
        long x = 0;
        long y = 0;
        long minX = Long.MAX_VALUE;
        long minY = Long.MAX_VALUE;
        long prefix = 0;
        for (int j = 0; j < n; j++) {
          if (j % 2 == i) {
            if (j > 0) {
              result = Math.min(result, (n - x) * a[j] + (n - y) * minY + prefix);
            }
            ++x;
            prefix += a[j];
            minX = Math.min(minX, a[j]);
          } else {
            if (j > 0) {
              result = Math.min(result, (n - y) * a[j] + (n - x) * minX + prefix);
            }
            ++y;
            prefix += a[j];
            minY = Math.min(minY, a[j]);
          }
        }
      }
      out.println(result);
    }

    out.close();
  }

  

  

  

  
}

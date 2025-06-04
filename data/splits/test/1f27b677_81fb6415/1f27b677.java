import java.util.*;
import java.io.*;

public class Codeforces {

  static long accurateFloor(long a, long b) {
    long val = a/b;
    while(val*b > a) val--;
    return val;
  }

  public static void main(String[] args) throws Exception {
      BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
      PrintWriter writer = new PrintWriter(System.out);
      int T = Integer.parseInt(br.readLine());
      while(T-- > 0) {
        StringTokenizer st = new StringTokenizer(br.readLine());
        int n = Integer.parseInt(st.nextToken());
        long k = Long.parseLong(st.nextToken());
        Long[] a = new Long[n];
        st = new StringTokenizer(br.readLine());
        long[] sum = new long[n];
        for(int i = 0; i < n; i++) {
          a[i] = Long.parseLong(st.nextToken());
        }
        Arrays.sort(a);
        sum[0] = a[0];
        for(int i = 1; i < n; i++) {
          sum[i] += sum[i-1]+a[i];
        }
        long ans = Long.MAX_VALUE;
        for(int y = 0; y < n; y++) {
          // long z = k-sum[n-y-1]+a[0];
          // if(z > 0) z = z/(y+1);
          // else {
          //   if(z%(y+1) == 0) z = z/(y+1);
          //   else {
          //     z = z/(y+1);
          //     z -= 1;
          //   }
          // }
          long x = a[0]-accurateFloor(k-sum[n-y-1]+a[0], y+1);
          x = Math.max(0, x);
          ans = Math.min(ans, y+x);
        }
        writer.println(ans);
      }
      writer.close();
      br.close();
  }

}

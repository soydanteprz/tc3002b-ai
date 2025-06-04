import java.util.*;
import java.io.*;

public class Codeforces {

  static int n;
  static int[] h = new int[200001];

  static boolean check(int x) {
    int[] cur_h = new int[n];
    for(int i = 0; i < n; i++) cur_h[i] = h[i];
    for(int i = n-1; i >= 2; i--) {
      if(cur_h[i] < x) return false;
      int d = Math.min(h[i], cur_h[i]-x)/3;
      cur_h[i-1] += d;
      cur_h[i-2] += 2*d;
    }
    return cur_h[0] >= x && cur_h[1] >= x;
  }

  public static void main(String[] args) throws Exception {
      BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
      PrintWriter writer = new PrintWriter(System.out);
      int T = Integer.parseInt(br.readLine());
      while(T-- > 0) {
        n = Integer.parseInt(br.readLine());
        StringTokenizer st = new StringTokenizer(br.readLine());
        int max = Integer.MIN_VALUE;
        for(int i = 0; i < n; i++) {
          h[i] = Integer.parseInt(st.nextToken());
          max = Math.max(max, h[i]);
        }
        int l = 0;
        int r = max;
        while(l < r) {
          int mid = l + (r-l+1)/2;
          if(check(mid)) l = mid;
          else r = mid-1;
        }
        writer.println(l);
      }
      writer.close();
      br.close();
  }

}

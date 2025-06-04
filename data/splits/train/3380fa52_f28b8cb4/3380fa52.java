import java.util.*;
import java.io.*;
import java.math.*;

/* Name of the class has to be "Main" only if the class is public. */
public class Coder {
  static int n, k;
  static long a[];
  static int pos[];
  static int temp[];
  static StringBuilder str = new StringBuilder("");
  static int cnt[][] = new int[(int)1e5+5][2];
  static void solve() {
    long []l = new long[n];
    long []r = new long[n];
    long p = Integer.MAX_VALUE;
    for(int i=0;i<n;i++){
        p = Math.min(p+1, a[i]);
        l[i] = p;
    }
    p=Integer.MAX_VALUE;
    for(int i=n-1;i>=0;i--){
     p = Math.min(p+1, a[i]);
     r[i] = p;
    }
    for(int i=0;i<n;i++){
        str.append(Math.min(l[i],r[i])).append(" ");
    }
    str.append("\n");
  }

  public static void main(String[] args) throws java.lang.Exception {
    BufferedReader bf = new BufferedReader(new InputStreamReader(System.in));
    int t = Integer.parseInt(bf.readLine().trim());
    while (t-- > 0) {
      bf.readLine();
      String s[] = bf.readLine().trim().split("\\s+");
      n = Integer.parseInt(s[0]);
      k = Integer.parseInt(s[1]);
      s = bf.readLine().trim().split("\\s+");
      pos = new int[k];
      temp = new int[k];
      a = new long[n];
      for(int i=0;i<k;i++) pos[i]=Integer.parseInt(s[i]);
      s = bf.readLine().trim().split("\\s+");
      for(int i=0;i<k;i++) temp[i]=Integer.parseInt(s[i]);
      Arrays.fill(a, Integer.MAX_VALUE);
      for(int i=0;i<k;i++) a[pos[i]-1] = temp[i]; 
      solve();
    }
    System.out.print(str);
  }
}

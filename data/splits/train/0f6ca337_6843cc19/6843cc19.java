import java.io.*;
import java.util.*;
 
import java.math.*;
import java.awt.Point;
 
public class Main {
    static final long MOD = 1000000007L;
    //static final long MOD2 = 1000000009L;
    //static final long MOD = 998244353L;
    //static final long INF = 500000000000L;
    static final int INF =   1000000005;
    static final int NINF = -1000000005;
    //static final long INF = 1000000000000000000L;
    static FastScanner sc;
    static PrintWriter pw;
    static final int[][] dirs = {{-1,0},{1,0},{0,-1},{0,1}};
 
    public static void main(String[] args) {
        sc = new FastScanner();
        pw = new PrintWriter(System.out);
 
 
        int Q = sc.ni();
        for (int q = 0; q < Q; q++) {
            int N = sc.ni();
            int[] nums = sc.intArray(N,0);
            Integer[] less = new Integer[N];
            Integer[] more = new Integer[N];
            TreeSet<Integer> ts = new TreeSet<Integer>();
            for (int i = 0; i < N; i++) {
                ts.add(nums[i]);
                less[i] = ts.lower(nums[i]);
                more[i] = ts.higher(nums[i]);
            }
            
            String ans = "YES";
            for (int i = 1; i < N; i++) {
                if (nums[i-1]==nums[i]) continue;
                
                if (nums[i-1] < nums[i]) {
                    if (more[i-1]!=null && more[i-1] < nums[i]) {
                        ans = "NO";
                        break;
                    }
                } else {
                    if (less[i-1]!=null && less[i-1] > nums[i]) {
                        ans = "NO";
                        break;
                    }
                }
            }
            pw.println(ans);
        }
        pw.close(); 
    }
    
 
    
 
    
 
    //Sort an array (immune to quicksort TLE)
    public static void sort(int[][] arr) {
        Random rgen = new Random();
        for (int i = 0; i < arr.length; i++) {
            int r = rgen.nextInt(arr.length);
            int[] temp = arr[i];
            arr[i] = arr[r];
            arr[r] = temp;
        }
        Arrays.sort(arr, new Comparator<int[]>() {
            
        });
    }
    
    public static void sort(long[][] arr) {
        Random rgen = new Random();
        for (int i = 0; i < arr.length; i++) {
            int r = rgen.nextInt(arr.length);
            long[] temp = arr[i];
            arr[i] = arr[r];
            arr[r] = temp;
        }
        Arrays.sort(arr, new Comparator<long[]>() {
            
        });
    }
 
    
}

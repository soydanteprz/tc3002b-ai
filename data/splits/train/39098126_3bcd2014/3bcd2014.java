import java.io.*;
import java.util.*;

import javax.sound.midi.MidiChannel;

public class Main {
    static PrintWriter out;
    static FastReader sc;

    public static void main(String[] args) throws FileNotFoundException {
        boolean t = true;
        boolean f = false;
        if (f) {
            out = new PrintWriter("output.txt");
            sc = new FastReader("input.txt");
        } else {
            out = new PrintWriter((System.out));
            sc = new FastReader();
        }

        int tt = 1;
        tt = sc.nextInt();
        while (tt-- > 0) {
            solve();
        }
        out.flush();
        out.close();
    }

    static boolean bs(int[] arr, int a){
    
      // out.println(" mid1 "  + Arrays.toString(arr) + " " + a);

      int[] curr = Arrays.copyOf(arr, arr.length);
     for(int i = arr.length - 1; i >= 2; i--){
       if(arr[i] < a){
         return false;
       }
       int min = Math.min(arr[i] - a, curr[i]);
       min = min/3;
       arr[i] -= 3*min;
       arr[i - 1] += min;
       arr[i-2] += 2*min;
     }

      for(int i: arr){
        if(i < a){
          return false;
        }
      }
      return true;
    }
    public static void solve() {
       int n = sc.nextInt();
       int[] arr = new int[n];
       int max = Integer.MIN_VALUE;
       for(int i = 0; i < n; i++){
         arr[i] = sc.nextInt();
         max = Math.max(max, arr[i]);
       }

       int l = 0, r = max, ans = 0;
       while(l <= r){
         int mid = (l+r)/2;
        //  out.println(" l r " + l + " " + r);
         int[] aa = Arrays.copyOf(arr, arr.length);
         if(bs(aa, mid)){
          // out.println(Arrays.toString(arr) + " " + Arrays.toString(aa) + " mid "  + mid );
            ans = mid;
            l = mid+1;
         }else{
           r = mid-1;
         }
       }
       out.println(ans);
    }

    

    

    

    

}

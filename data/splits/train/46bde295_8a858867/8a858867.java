import java.io.*;
import java.util.*;
 
 
public class Mainnn{
    // static final File ip = new File("input.txt");
    // static final File op = new File("output.txt");
    // static {
    //     try {
    //         System.setOut(new PrintStream(op));
    //         System.setIn(new FileInputStream(ip));
    //     } catch (Exception e) {
    //     }

    //     // in = new InputReader(System.in);
    // }



   public static void main(String[] args) {
      MyScanner sc = new MyScanner();
      out = new PrintWriter(new BufferedOutputStream(System.out));
      
      //out.println(result);                    // print via PrintWriter
      /////////////////////////////////////////////

        int test = sc.nextInt();
        while(test-- != 0)
        {
            int n = sc.nextInt();
            String[] info = new String[n];
            for (int i = 0; i < n; i++) {
                info[i] = sc.next();
            }
            int ans = 0;
            for (int i = 0; i < 5; i++) {
                char cur = (char) (i+'a');
                int[] map = new int[n];
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < info[j].length(); k++) {
                        if(info[j].charAt(k)==cur)map[j]++;
                        else map[j]--;
                    }
                }
                Arrays.sort(map);
                int sum = 0;
                for(int j = n-1; j >= 0; j--) {
                    sum += map[j];
                    if(sum<=0){
                        break;
                    }
                    ans = Math.max(ans,n-j);
                }
            }
            System.out.println(ans);
        }





      /////////////////////////////////////////////
      // Stop writing your solution here. -------------------------------------
      out.close();
   }
   
   

   //-----------PrintWriter for faster output---------------------------------
   public static PrintWriter out;
      
   //-----------MyScanner class for faster input----------
   
   //--------------------------------------------------------
}

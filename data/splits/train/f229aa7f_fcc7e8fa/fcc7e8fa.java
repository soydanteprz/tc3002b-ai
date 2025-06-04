import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.*;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.InputStream;

public class Main {

    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskA solver = new TaskA();
        int t;
        t = in.nextInt();
        //t = 1;
        while (t > 0) {
            solver.call(in,out);
            t--;
        }
        out.close();
    }

    static class TaskA {
        public void call(InputReader in, PrintWriter out) {
            int n, _00 = 0, _01 = 0, _11 = 0, _10 = 0;
            n = in.nextInt();
            char[] s = in.next().toCharArray();
            char[] s1 = in.next().toCharArray();

            for (int i = 0; i < n; i++) {
                if(s[i]==s1[i]){
                    if(s[i]=='0'){
                        _00++;
                    }
                    else{
                        _11++;
                    }
                }
                else{
                    if(s[i]=='0'){
                        _01++;
                    }
                    else{
                        _10++;
                    }
                }
            }
            int ans = Integer.MAX_VALUE;

            if(_10 ==_01){
                ans = 2*_01;
            }
            if(_11 == _00 + 1){
                ans = Math.min(ans, 2*_00 + 1);
            }

            if(ans == Integer.MAX_VALUE){
                out.println(-1);
            }
            else{
                out.println(ans);
            }
        }
    }

    

    

    

    

    

    

    static final Random random=new Random();

    

    
}

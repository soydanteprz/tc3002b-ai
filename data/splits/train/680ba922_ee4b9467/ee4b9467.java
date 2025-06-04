import java.io.*;
import java.util.*;
 
public class _1615c {
    FastScanner scn;
    PrintWriter w;
    PrintStream fs;

    long MOD = 1000000007;
    int MAX = 200005;
    
    
    
    
    boolean LOCAL;
    
    //SUFFICIENT DRY RUN????LOGIC VERIFIED FOR ALL TEST CASES???
    
    void solve(){
        
        int t=scn.nextInt();
            while(t-->0)
            {
                int n=scn.nextInt();
                String s1= scn.next();
                String s2 = scn.next();

                int op1=0,op0=0,one=0,zr=0;
                for(int i=0;i<n;i++){
                    char ch1 = s1.charAt(i);
                    char ch2 = s2.charAt(i);
                    
                    if(ch1=='1'&&ch2=='0'){
                        op1++;
                    }else if(ch1=='0'&&ch2=='1'){
                        op0++;
                    }else if(ch1==ch2){
                        if(ch1=='0'){
                            zr++;
                        }else{
                            one++;
                        }
                    }
                }
                
                if((one-zr)==1&&op1==op0){
                    w.println(Math.min(2*op1,one+zr));
                }else if((one-zr)==1){
                    w.println(one+zr);
                }else if(op1==op0){
                    w.println(2*op1);
                }else{
                    w.println(-1);
                }
            
            
            }
        
    }
 
    void run() {
        try {
            long ct = System.currentTimeMillis();
            scn = new FastScanner(new File("input.txt"));
            w = new PrintWriter(new File("output.txt"));
            fs=new PrintStream("error.txt");
            System.setErr(fs);
            LOCAL=true;
 
            solve();
 
            w.close();
            System.err.println(System.currentTimeMillis() - ct);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
 
    void runIO() {
 
        scn = new FastScanner(System.in);
        w = new PrintWriter(System.out);
        LOCAL=false;
 
        solve();
 
        w.close();
    }
 
    
 
    
    
    
    
    
     // TC- O(logmax(a,b))
    
 
    public static void main(String[] args) {
        new _1615c().runIO();
    }
}

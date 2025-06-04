import java.util.*;

// import jdk.tools.jlink.internal.SymLinkResourcePoolEntry;

// 
// import jdk.tools.jlink.internal.SymLinkResourcePoolEntry;

// import org.graalvm.compiler.asm.sparc.SPARCAssembler.Asi;

import java.io.*;

public class Main{

    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        
        // Start writing your solution here. -------------------------------------
        /*
        int n      = sc.nextInt();        // read input as integer
        long k     = sc.nextLong();       // read input as long
        double d   = sc.nextDouble();     // read input as double
        char c = sc.next().charAt(0);     // read input as char
        String str = sc.next();           // read input as String
        String s   = sc.nextLine();       // read whole line as String
        out.println(result);              // print via PrintWriter
        */
        //------------------------------------------------------------------

        int test = sc.nextInt();
        while(test -- > 0){

            int n = sc.nextInt();
            List<Integer> a = new ArrayList<>();
            List<Integer> b = new ArrayList<>();
            List<Integer> c = new ArrayList<>();
            List<Integer> d = new ArrayList<>();
            List<Integer> e = new ArrayList<>();

            while(n-->0){
                int[] arr = new int[5];
                // System.out.println(Arrays.toString(arr));

                char[] ch = sc.nextLine().toCharArray();
                int len = ch.length;

                for(char chhh : ch){
                    arr[chhh-'a']++;
                }

                for(int i = 0 ; i < 5 ; i++){
                    arr[i] = 2*arr[i] - len;
                }

                // System.out.println(Arrays.toString(arr));

                a.add(arr[0]);
                b.add(arr[1]);
                c.add(arr[2]);
                d.add(arr[3]);
                e.add(arr[4]);  
            }

            
            
            Collections.sort(a);
            Collections.sort(b);
            Collections.sort(c);
            Collections.sort(d);
            Collections.sort(e);

            // System.out.println(a);
            // System.out.println(b);
            // System.out.println(c);
            // System.out.println(d);
            // System.out.println(e);

            int max = 0;
            int currSum = 0;
            int words = 0;

            boolean flag = true;
            for(int i = a.size()-1 ; i >= 0 ; i--){
                if(currSum + a.get(i) <= 0){
                    max = Math.max(max , words);
                    flag = false;
                    break;
                }else{
                    currSum += a.get(i);
                }
                words++;
            }

            if(flag)
                max = Math.max(a.size(), max);


            //end mein check haina???

            // System.out.println(max);
            

            currSum = 0;
            words = 0;
            flag = true;
            for(int i = b.size()-1 ; i >= 0 ; i--){
                if(currSum + b.get(i) <= 0){
                    max = Math.max(max , words);
                    flag = false;
                    break;
                }else{
                    currSum += b.get(i);
                }
                words++;
            }

            if(flag)
                max = Math.max(a.size(), max);
            
            
            currSum = 0;
            words = 0;
            flag = true;
            for(int i = c.size()-1 ; i >= 0 ; i--){
                if(currSum + c.get(i) <= 0){
                    max = Math.max(max , words);
                    flag = false;
                    break;
                }else{
                    currSum += c.get(i);
                }
                words++;
            }

    
            if(flag)
                max = Math.max(a.size(), max);
            
            currSum = 0;
            words = 0;
            flag = true;
            for(int i = d.size()-1 ; i >= 0 ; i--){
                if(currSum + d.get(i) <= 0){
                    max = Math.max(max , words);
                    flag = false;
                    break;
                }else{
                    currSum += d.get(i);
                }
                words++;
            }

            if(flag)
                max = Math.max(a.size(), max);

            
            currSum = 0;
            words = 0;
            flag = true;
            for(int i = e.size()-1 ; i >= 0 ; i--){
                if(currSum + e.get(i) <= 0){
                    max = Math.max(max , words);
                    flag = false;
                    break;
                }else{
                    currSum += e.get(i);
                }
                words++;
            }
            
            if(flag)
                max = Math.max(a.size(), max);

            out.println(max);

        }

        //------------------------------------------------------------------
        
        // Stop writing your solution here. -------------------------------------
        out.close();
    }

    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------

    //for calculating binomialCoeff
    

    //Pair with int int 
    

    //Shortcut function
    

    //let's make one for calculating lcm basically
    

    //int version for gcd
    

    //long version for gcd
    

    //swapping two elements in an array
    

    //reversing an array
    

     
    //-----------PrintWriter for faster output---------------------------------
    public static PrintWriter out;
      
    //-----------MyScanner class for faster input----------
    
   //--------------------------------------------------------
}
